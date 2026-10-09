"""Keep the publishable working tree portable and free of manuscript revision names."""

from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tomllib
import unittest
from unittest.mock import call, patch
from urllib.parse import urlsplit

import yaml


REPO = Path(__file__).resolve().parents[2]
RETIRED_NAMES = ("Graph" + "Decide", "Jev" + "GraphBench")
PUBLIC_ROOT_FILES = frozenset({
    ".gitignore", "CITATION.cff", "README.md", "pyproject.toml", "run_benchmark.py",
    "data/README.md", "data/manifest.json", "data/optimization-catalog.json",
})
PUBLIC_DIRECTORIES = (
    "src/", "tests/", "configs/", "docs/", "assets/benchmark/", "data/results/",
)
PUBLIC_EXTENSIONS = frozenset({".py", ".md", ".json", ".yaml", ".yml", ".csv", ".svg"})
LOCAL_DIRECTORIES = frozenset({
    "output", "build", "dist", "archive", ".git", ".cache", ".test-work",
    "__pycache__", ".venv", "venv", "node_modules",
})
SENSITIVE_FILENAME = re.compile(
    r"(?:^|[._-])(?:private|credentials?|secrets?|tokens?|passwords?|api[_-]?keys?|"
    r"env|environment|startup|id_rsa|id_dsa|id_ecdsa|id_ed25519|"
    r"bashrc|zshrc|bash_profile|bash_login|bash_logout|zprofile|zlogin|zshenv|"
    r"profile|login|netrc|npmrc|pypirc|direnv|ssh|aws|azure|gnupg|"
    r"activate|pythonstartup|bash_history|zsh_history)(?:$|[._-])|^\.env|"
    r"\.(?:pem|key|p12|pfx|keystore)$", re.I,
)
INTERNAL_REVISION = re.compile(
    # Date-stamped experiments and explicitly named manuscript revisions have
    # no revision-number range. Bare historical manuscript labels are retained
    # as a separate check; embedded model IDs and dotted releases are not labels.
    r"(?<![A-Za-z0-9])(?:19|20)[0-9]{2}[-_]?[0-9]{2}[-_]?[0-9]{2}"
    r"[\s_-]*(?:v|version[\s_-]*)[0-9]+(?![0-9]|\.[0-9])"
    r"|(?<![A-Za-z0-9])(?:paper|manuscript|draft|edition|revision|verified-main)"
    r"[\s_-]+(?:v|version[\s_-]*)[0-9]+(?![0-9]|\.[0-9])"
    r"|(?<![A-Za-z])(?:v|version[\s_-]*)(?:2[0-9]|3[0-9])"
    r"(?![0-9]|\.[0-9])", re.I,
)
CJK = re.compile(
    r"[\u1100-\u11ff\u2e80-\u2fff\u3000-\u31ff\u3400-\u4dbf\u4e00-\u9fff"
    r"\ua960-\ua97f\uac00-\ud7ff\uf900-\ufaff\uff66-\uff9f"
    r"\U0001b000-\U0001b2ff\U00020000-\U000323af]"
)
PERSONAL_HOME = re.compile(r"/(?:home|Users|fastdata)/[A-Za-z0-9][^\s\"']*")


def has_retired_name(text):
    return any(name.lower() in text.lower() for name in RETIRED_NAMES)


def public_path_problem(relative):
    """Classify names before any content read; unrecognized paths fail closed."""
    parts = PurePosixPath(relative).parts
    if (not parts or relative.startswith("/") or "\\" in relative
            or any(part in {".", ".."} for part in relative.split("/"))
            or PurePosixPath(relative).as_posix() != relative):
        return "Invalid repository-relative path"
    if any(SENSITIVE_FILENAME.search(part) for part in parts):
        return "Sensitive filename"
    if any(part.casefold() in LOCAL_DIRECTORIES for part in parts[:-1]):
        return "Local-only directory"
    if relative in PUBLIC_ROOT_FILES:
        return None
    if not relative.startswith(PUBLIC_DIRECTORIES):
        return "Nonpublic path"
    if PurePosixPath(relative).suffix not in PUBLIC_EXTENSIONS:
        return "Unapproved public extension"
    return None


def public_file(root, relative):
    problem = public_path_problem(relative)
    if problem:
        raise ValueError(problem)
    path = root
    # Check ancestors too: an ordinary-looking file beneath a linked directory
    # must never cause content outside the public tree to be opened.
    for part in PurePosixPath(relative).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Symlink in public path")
    return path


def read_public_text(relative):
    return public_file(REPO, relative).read_text(encoding="utf-8")


def identity_pattern(citation):
    names = [urlsplit(citation["repository-code"]).path.strip("/").split("/")[0]]
    for author in citation["authors"]:
        given, family = author["given-names"], author["family-names"]
        # Short given names also occur as surnames in scientific references and
        # as markup/code tokens. Match those only as a complete author name.
        if len(given) > 3:
            names.append(given)
        names.extend((f"{given} {family}", f"{family}, {given}"))
    return re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(map(re.escape, names))
                      + r")(?![A-Za-z0-9])", re.I)


def citation_author_spans(text, citation):
    """Allow only known scalar author values at the two intentional CFF paths."""
    try:
        document = yaml.compose(text)
    except yaml.YAMLError:
        return {}

    def field(node, key):
        if isinstance(node, yaml.MappingNode):
            matches = [(name, value) for name, value in node.value
                       if isinstance(name, yaml.ScalarNode) and name.value == key]
            if len(matches) == 1:
                name, value = matches[0]
                # An alias points to an earlier definition, potentially outside
                # the author fields. Never exempt that unrelated source line.
                if value.start_mark.index >= name.end_mark.index:
                    return value
        return None

    spans = {}
    for parent in (document, field(document, "preferred-citation")):
        authors = field(parent, "authors")
        if not isinstance(authors, yaml.SequenceNode):
            continue
        for author in authors.value:
            if not isinstance(author, yaml.MappingNode) or len(author.value) != 2:
                continue
            values = {key: field(author, key) for key in ("given-names", "family-names")}
            if not all(isinstance(value, yaml.ScalarNode) and value.tag == "tag:yaml.org,2002:str"
                       for value in values.values()):
                continue
            if {key: value.value for key, value in values.items()} not in citation["authors"]:
                continue
            for value in values.values():
                start, end = value.start_mark, value.end_mark
                if start.line == end.line:
                    # YAML aliases can reference the same scalar twice. Removing
                    # a span twice could accidentally hide an adjacent comment.
                    spans.setdefault(start.line + 1, set()).add((start.column, end.column))
    return spans


def readme_author_field(line, citation):
    """Recognize a complete BibTeX author field, not notes or arbitrary credit."""
    match = re.fullmatch(r'\s*author\s*=\s*(?:\{([^{}]*)\}|"([^"{}]*)")\s*,?\s*', line, re.I)
    if not match:
        return False
    names = re.split(r"\s+and\s+", (match[1] or match[2] or "").strip())
    expected = citation["authors"]
    return len(names) == len(expected) and all(
        " ".join(name.split()) in (f"{author['family-names']}, {author['given-names']}",
                                   f"{author['given-names']} {author['family-names']}")
        for name, author in zip(names, expected)
    )


def text_problems(relative, text, citation):
    identity = identity_pattern(citation)
    repository_url = re.compile(
        r"(?<![A-Za-z0-9_./:-])" + re.escape(citation["repository-code"])
        + r"(?![A-Za-z0-9_/?#.%+@=~:-])"
    )
    author_spans = citation_author_spans(text, citation) if relative == "CITATION.cff" else {}
    in_citation = in_bibtex = in_entry = False
    errors = []
    for number, line in enumerate(text.splitlines(), 1):
        if INTERNAL_REVISION.search(line) or PERSONAL_HOME.search(line) or has_retired_name(line):
            errors.append((relative, number, "Internal name or personal home path"))
        if CJK.search(line):
            errors.append((relative, number, "CJK text in public content"))
        if relative == "README.md":
            if re.match(r"^#{1,6}\s+", line):
                in_citation = line.strip().casefold() == "## citation"
                in_bibtex = in_entry = False
            if line.strip().casefold() == "```bibtex" and in_citation:
                in_bibtex = True
            elif line.strip() == "```":
                in_bibtex = in_entry = False
            if in_bibtex and re.fullmatch(
                    r"\s*@(?:article|book|booklet|conference|inbook|incollection|inproceedings|"
                    r"manual|mastersthesis|misc|phdthesis|proceedings|software|techreport|unpublished)"
                    r"\{[^\s{},]+,\s*", line, re.I):
                in_entry = True
            elif line.strip() == "}":
                in_entry = False
        identity_line = line
        for start, end in sorted(author_spans.get(number, ()), reverse=True):
            identity_line = identity_line[:start] + identity_line[end:]
        if in_bibtex and in_entry and readme_author_field(line, citation):
            identity_line = ""
        # Only the exact repository URL grants an owner exception, not arbitrary
        # owner mentions, profile URLs, URL prefixes or URLs to other projects.
        if identity.search(repository_url.sub("", identity_line)):
            errors.append((relative, number, "Personal attribution outside citation material"))
    return errors


def scan_public_paths(root, paths, citation):
    errors = []
    identity = identity_pattern(citation)
    for relative in sorted(set(paths)):
        if not relative:  # Git's NUL-delimited output ends with an empty entry.
            continue
        try:
            path = public_file(root, relative)
        except ValueError as exc:
            errors.append((relative, str(exc)))
            continue
        if not path.exists():  # Deleted index entries have no working-tree content.
            continue
        if not path.is_file():
            errors.append((relative, "Not a regular public file"))
            continue
        if (INTERNAL_REVISION.search(relative) or has_retired_name(relative)
                or identity.search(relative)):
            errors.append((relative, "Internal or personal filename"))
            continue
        if CJK.search(relative):
            errors.append((relative, "CJK public filename"))
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeError, OSError):
            errors.append((relative, "Public text is unreadable or not UTF-8"))
            continue
        errors.extend(text_problems(relative, text, citation))
    return errors


class PublicRepositoryTests(unittest.TestCase):
    def test_citation_preserves_public_attribution(self):
        citation = yaml.safe_load(read_public_text("CITATION.cff"))
        self.assertEqual(citation["cff-version"], "1.2.0")
        self.assertEqual(citation["type"], "software")
        self.assertEqual(citation["title"], "GraphDecisionBench")
        self.assertIn("GraphDecisionBench", citation["message"])
        # Public author credit is intentional, unlike private machine paths.
        expected_authors = [
            {"family-names": "Ya" + "ng", "given-names": "Xian" + "liang"},
            {"family-names": "Zh" + "ang", "given-names": "Ya" + "pu"},
            {"family-names": "Zh" + "ao", "given-names": "L" + "i"},
        ]
        self.assertEqual(citation["authors"], expected_authors)
        repository_url = citation["repository-code"]
        parsed = urlsplit(repository_url)
        self.assertEqual(parsed.scheme, "https")
        self.assertEqual(parsed.netloc, "github.com")
        self.assertEqual(parsed.query, "")
        self.assertEqual(parsed.fragment, "")
        owner, project = parsed.path.removeprefix("/").split("/")
        self.assertRegex(owner, r"^[A-Za-z0-9][A-Za-z0-9-]*$")
        self.assertEqual(project, "GraphDecisionBench")
        self.assertFalse(has_retired_name(repository_url))
        self.assertEqual(citation["url"], citation["repository-code"])
        preferred = citation["preferred-citation"]
        self.assertEqual(preferred["type"], "article")
        self.assertEqual(preferred["title"], "GraphDecisionBench: Benchmarking System One Models on Graph Tasks")
        self.assertEqual(preferred["authors"], citation["authors"])
        self.assertEqual(preferred["url"], "https://arxiv.org/abs/2610.06354")
        self.assertEqual(preferred["journal"], "arXiv")
        self.assertEqual(preferred["doi"], "10.48550/arXiv.2610.06354")
        self.assertEqual(preferred["year"], 2026)
        readme = read_public_text("README.md")
        self.assertIn(preferred["title"], readme)
        self.assertIn(preferred["url"], readme)
        self.assertIn(preferred["doi"], readme)
        self.assertIn("eprint       = {2610.06354}", readme)
        self.assertNotIn("howpublished = {Software}", readme)
        self.assertIn(citation["title"], read_public_text("README.md"))
        self.assertIn(repository_url, read_public_text("README.md"))

    def test_public_metadata_uses_project_identity(self):
        data = json.loads(read_public_text("data/results/summary.json"))
        self.assertEqual(data["benchmark"], "GraphDecisionBench")
        project = tomllib.loads(read_public_text("pyproject.toml"))["project"]
        self.assertEqual(project["name"], "GraphDecisionBench")
        self.assertEqual(project["name"].lower(), "graphdecisionbench")
        from src.utils import public_graphtext, public_optimization, public_structural
        for module, suffix in ((public_graphtext, "graphtext"), (public_optimization, "optimization"),
                               (public_structural, "structural")):
            self.assertEqual(module.PROTOCOL, "GraphDecisionBench-public-" + suffix)

    def test_retired_name_guard_rejects_all_spellings_including_old_urls(self):
        repository_url = yaml.safe_load(read_public_text("CITATION.cff"))["repository-code"]
        for name in RETIRED_NAMES:
            for spelling in (name, name.lower(), name.upper()):
                with self.subTest(name=spelling):
                    self.assertTrue(has_retired_name(spelling))
                    self.assertTrue(has_retired_name(spelling + "_SCORING_MODEL"))
                    old_url = repository_url.rsplit("/", 1)[0] + "/" + spelling
                    self.assertTrue(has_retired_name('url: "' + old_url + '"'))
                    self.assertTrue(has_retired_name(old_url + "/tree/main"))
        self.assertFalse(has_retired_name('url: "' + repository_url + '"'))

    def test_public_results_have_one_canonical_home(self):
        expected = {"README.md", "summary.json"}
        directory = REPO / "data/results"
        self.assertEqual({path.name for path in directory.iterdir()}, expected)
        self.assertEqual({path.name for path in (REPO / "assets/benchmark").iterdir()},
                         {"overview.svg", "structural-queries.svg", "graph-text-decisions.svg",
                          "optimization-trajectories.svg"})
        if shutil.which("git") and (REPO / ".git").exists():
            ignored = subprocess.run(["git", "check-ignore", "--no-index", *[
                f"data/results/{name}" for name in sorted(expected)]],
                cwd=REPO, capture_output=True, text=True, check=False)
            self.assertEqual(ignored.returncode, 1, ignored.stdout + ignored.stderr)

    @unittest.skipUnless(shutil.which("git"), "Git is required to enumerate the public working tree")
    def test_public_paths_and_text_have_no_internal_revisions_or_personal_home_paths(self):
        process = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                                 cwd=REPO, check=False, capture_output=True)
        if process.returncode:
            self.skipTest("No Git checkout available")
        paths = sorted(set(process.stdout.decode().split("\0")))
        citation = yaml.safe_load(read_public_text("CITATION.cff"))
        self.assertEqual(scan_public_paths(REPO, paths, citation), [])


class PublicScannerTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("/public-repository")
        self.owner = "Example" + "Maintainer"
        self.given = "Example" + "Given"
        self.citation = {
            "repository-code": "https://github.com/" + self.owner + "/GraphDecisionBench",
            "authors": [
                {"given-names": self.given, "family-names": "Researcher"},
                {"given-names": "L" + "i", "family-names": "Zh" + "ao"},
            ],
        }

    def scan_fake(self, paths, contents=None, links=(), read_error=None):
        """Exercise the real scanner without opening or creating any fixture files."""
        contents = contents or {}

        def read(path, encoding):
            self.assertEqual(encoding, "utf-8")
            relative = path.relative_to(self.root).as_posix()
            self.assertIn(relative, contents, "An unapproved fixture was opened")
            if read_error:
                raise read_error
            return contents[relative]

        with patch.object(Path, "read_text", autospec=True, side_effect=read) as reader, \
                patch.object(Path, "exists", return_value=True), \
                patch.object(Path, "is_file", return_value=True), \
                patch.object(Path, "is_symlink", autospec=True,
                             side_effect=lambda path: path in {self.root / p for p in links}):
            errors = scan_public_paths(self.root, paths, self.citation)
        return errors, reader

    def test_sensitive_and_nonpublic_paths_are_never_read(self):
        names = (
            ".env", ".env.local", ".envrc", "environment.yml", "startup.py",
            ".bashrc", ".bash_profile", ".zshenv", ".profile", ".netrc",
            "credentials.json", "service.token.json", "api-key.yaml", "private/data.json",
            ".ssh/id_ed25519", "server.pem", "activate.py",
        )
        paths = [prefix + name for prefix in ("", *PUBLIC_DIRECTORIES) for name in names]
        paths.extend(("output/report.md", "build/lib/module.py", "dist/package.json",
                      "repo/notes.md", "results/summary.json", "data/raw/graph.json",
                      "assets/other/figure.svg", "local/notes.md", "NOTICE.md"))
        paths.extend(prefix + directory + "/notes.md"
                 for prefix in PUBLIC_DIRECTORIES for directory in LOCAL_DIRECTORIES)
        errors, reader = self.scan_fake(paths)
        reader.assert_not_called()
        self.assertEqual({item[0] for item in errors}, set(paths))

    def test_symlink_files_and_ancestors_are_never_read(self):
        paths = ["docs/linked.md", "src/linked/module.py", "README.md"]
        errors, reader = self.scan_fake(paths, links=("docs/linked.md", "src", "README.md"))
        reader.assert_not_called()
        self.assertEqual({item[0] for item in errors}, set(paths))
        self.assertTrue(all(item[1] == "Symlink in public path" for item in errors))

    def test_unknown_public_files_fail_closed_without_reading(self):
        paths = [prefix + name for prefix in PUBLIC_DIRECTORIES
                 for name in ("unknown.bin", "no-extension", "notes.md.bak")]
        errors, reader = self.scan_fake(paths)
        reader.assert_not_called()
        self.assertEqual({item[0] for item in errors}, set(paths))
        self.assertTrue(all(item[1] == "Unapproved public extension" for item in errors))

    def test_absolute_and_traversal_paths_are_never_read(self):
        paths = ["/" + "home" + "/example/notes.md", "../README.md",
                 "docs/../README.md", "docs//notes.md", "./README.md",
                 "docs\\notes.md", "docs/./notes.md"]
        errors, reader = self.scan_fake(paths)
        reader.assert_not_called()
        self.assertEqual({item[0] for item in errors}, set(paths))

    def test_allowlisted_files_are_read_even_when_new(self):
        paths = sorted(PUBLIC_ROOT_FILES) + [
            "src/new.py", "tests/new.py", "configs/new.yaml", "docs/new.md",
            "assets/benchmark/new.svg", "data/results/new.json",
        ]
        errors, reader = self.scan_fake(paths + ["", paths[0]],
                                        {path: "Public content." for path in paths})
        self.assertEqual(errors, [])
        self.assertEqual(reader.call_args_list,
                         [call(self.root / path, encoding="utf-8") for path in sorted(paths)])

    def test_unreadable_public_text_fails_closed(self):
        for error in (UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid"), OSError("unreadable")):
            with self.subTest(error=type(error).__name__):
                errors, reader = self.scan_fake(["docs/new.md"], {"docs/new.md": ""},
                                                read_error=error)
                reader.assert_called_once()
                self.assertEqual(errors, [("docs/new.md", "Public text is unreadable or not UTF-8")])

    def test_deleted_index_entries_are_not_read(self):
        with patch.object(Path, "is_symlink", return_value=False), \
                patch.object(Path, "exists", return_value=False), \
                patch.object(Path, "read_text") as reader:
            self.assertEqual(scan_public_paths(self.root, ["docs/deleted.md"], self.citation), [])
            reader.assert_not_called()

    def test_revision_guard_covers_bare_historical_manuscript_forms(self):
        for number in range(20, 40):
            for label in (f"v{number}", f"v{number}draft", f"v{number}_final",
                          f"version{number}", f"version_{number}", f"version {number}",
                          f"2026v{number}", f"2026-v{number}"):
                with self.subTest(label=label):
                    self.assertIsNotNone(INTERNAL_REVISION.search(label.upper()))
                    self.assertTrue(text_problems("docs/new.md", label, self.citation))
                    path = "docs/" + label + ".md"
                    errors, reader = self.scan_fake([path])
                    reader.assert_not_called()
                    self.assertEqual(errors, [(path, "Internal or personal filename")])
        for label in ("Decider2Bv11", "model v11", "version 1.2.0", "2026", "v200",
                      *(f"v{number}" for number in (19, 40, 99))):
            with self.subTest(label=label):
                self.assertIsNone(INTERNAL_REVISION.search(label))

    def test_dated_and_explicit_manuscript_revisions_have_no_number_range(self):
        for number in (1, 2, 3, 19, 20, 32, 33, 39, 40, 99, 200):
            for prefix in ("20261009-", "2026-10-09_", "20261009", "paper-",
                           "manuscript ", "draft_", "edition-", "revision-", "verified-main-"):
                for revision in (f"v{number}", f"version_{number}"):
                    label = prefix + revision
                    with self.subTest(label=label):
                        self.assertTrue(text_problems("docs/new.md", label, self.citation))
                        path = "docs/" + label.upper() + ".md"
                        errors, reader = self.scan_fake([path])
                        reader.assert_not_called()
                        self.assertEqual(errors, [(path, "Internal or personal filename")])

    def test_real_model_protocol_schema_and_dotted_software_versions_are_preserved(self):
        labels = ["Decider-2B v11", "Qwen3.5-4B", "Jev-1.13.0", "schema_version: 1",
                  "choice-v1", "public-full-input-v3", "algorithmic_v1", "https://example.org/v1"]
        for number in (1, 3, 20, 32, 33, 39, 40, 200):
            labels.extend((f"Decider2Bv{number}", f"v{number}.2.0",
                           f"version {number}.2.0", f"version v{number}.5.*"))
        for label in labels:
            with self.subTest(label=label):
                self.assertEqual(text_problems("docs/new.md", label, self.citation), [])

    def test_cjk_guard_checks_text_and_names_without_literal_cjk_fixtures(self):
        for codepoint in (0x4E2D, 0x6587, 0x3400, 0xF900, 0x20000, 0x30000,
                          0x3042, 0x30A2, 0xAC00):
            character = chr(codepoint)
            for relative in ("README.md", "CITATION.cff", "src/new.py", "tests/new.py",
                             "configs/new.yaml", "docs/new.md", "assets/benchmark/new.svg"):
                with self.subTest(codepoint=codepoint, relative=relative):
                    self.assertTrue(text_problems(relative, character, self.citation))
            path = "docs/" + character + ".md"
            errors, reader = self.scan_fake([path])
            reader.assert_not_called()
            self.assertEqual(errors, [(path, "CJK public filename")])
        self.assertEqual(text_problems("docs/new.md", "English, Greek: alpha = \u03b1; 95% \u00b1 1.",
                                       self.citation), [])

    def test_retired_and_personal_filenames_are_rejected_without_reads(self):
        paths = ["docs/" + name + ".md" for name in (*RETIRED_NAMES, self.given, self.owner)]
        errors, reader = self.scan_fake(paths)
        reader.assert_not_called()
        self.assertEqual({item[0] for item in errors}, set(paths))

    def test_personal_names_and_local_paths_are_rejected_across_public_text(self):
        for relative in ("README.md", "src/new.py", "tests/new.py", "configs/new.yaml",
                         "docs/new.md", "assets/benchmark/new.svg", "data/results/new.json"):
            for text in (self.given, self.owner, self.owner.lower(),
                         *("/" + base + "/example/project" for base in ("home", "Users", "fastdata"))):
                with self.subTest(relative=relative, text=text):
                    self.assertTrue(text_problems(relative, text, self.citation))

    def test_citation_exceptions_do_not_exempt_intro_or_other_sections(self):
        credit = " and ".join(f"{author['family-names']}, {author['given-names']}"
                              for author in self.citation["authors"])
        citation_block = ("## Citation\n```bibtex\n@misc{benchmark,\n  author = {"
                          + credit + "},\n}\n```\n")
        self.assertEqual(text_problems("README.md", citation_block, self.citation), [])
        self.assertTrue(text_problems("CITATION.cff", credit, self.citation))
        for text in (credit + "\n" + citation_block, citation_block + credit,
                     citation_block + "## More\n" + credit,
                     "## Citation\n" + credit,
                     "```bibtex\n" + credit + "\n```",
                     citation_block.replace("author =", "note ="),
                     citation_block.replace("author =", "title ="),
                     citation_block.replace("@misc", "@comment"),
                     citation_block.replace(credit, self.owner),
                     citation_block.replace("},\n}", "}, % " + self.owner + "\n}"),
                     citation_block.replace("@misc{benchmark,\n", "")):
            with self.subTest(text=text):
                self.assertTrue(text_problems("README.md", text, self.citation))
        for relative in ("docs/new.md", "assets/benchmark/new.svg", "data/results/README.md"):
            self.assertTrue(text_problems(relative, citation_block, self.citation))
        # Citation exceptions do not permit local machine paths or revision labels.
        for relative in ("README.md", "CITATION.cff"):
            for invalid in ("v" + str(30 + 3), "/" + "home" + "/example/project", RETIRED_NAMES[0]):
                self.assertTrue(text_problems(relative, citation_block.replace(credit, invalid), self.citation))

    def test_cff_exception_is_limited_to_known_structured_author_values(self):
        document = {**self.citation, "preferred-citation": {"authors": self.citation["authors"]}}
        text = yaml.safe_dump(document)
        self.assertEqual(text_problems("CITATION.cff", text, self.citation), [])
        for field in ("message", "title", "notes", "contact"):
            for parent in (document, document["preferred-citation"]):
                with self.subTest(field=field, nested=parent is not document):
                    altered = deepcopy(document)
                    target = altered if parent is document else altered["preferred-citation"]
                    target[field] = self.given + " " + self.owner
                    self.assertTrue(text_problems("CITATION.cff", yaml.safe_dump(altered), self.citation))
        for invalid in (text + "# " + self.owner,
                        text.replace(self.given, self.given + " # " + self.owner),
                        yaml.safe_dump({"notes": {"authors": self.citation["authors"]}}),
                        yaml.safe_dump({"notes": self.citation["authors"],
                                        "authors": self.citation["authors"]}, sort_keys=False),
                        yaml.safe_dump({"authors": [{"given-names": self.given,
                                                    "family-names": self.owner}]})):
            self.assertTrue(text_problems("CITATION.cff", invalid, self.citation))

    def test_only_canonical_repository_url_exempts_owner(self):
        url = self.citation["repository-code"]
        for relative in ("README.md", "docs/new.md", "assets/benchmark/new.svg", "src/new.py"):
            for text in (url, f"[Repository]({url})", f'<a href="{url}">Repository</a>'):
                self.assertEqual(text_problems(relative, text, self.citation), [])
            for text in (self.owner, url.rsplit("/", 1)[0], url + "-other", url + "/issues",
                         url + "?owner=" + self.owner, url.replace("https:", "http:"),
                         url.replace("github.com", "example.org"), url + " " + self.owner):
                with self.subTest(relative=relative, text=text):
                    self.assertTrue(text_problems(relative, text, self.citation))

    def test_scientific_reference_surnames_and_markup_are_not_personal_attribution(self):
        references = (self.citation["authors"][1]["given-names"]
                      + " et al. (2024); Smith et al. (2023). <li>Sources</li>")
        for relative in ("docs/new.md", "assets/benchmark/new.svg", "README.md"):
            self.assertEqual(text_problems(relative, references, self.citation), [])
        author = self.citation["authors"][1]
        self.assertTrue(text_problems("docs/new.md", author["given-names"] + " "
                                      + author["family-names"], self.citation))


if __name__ == "__main__":
    unittest.main()