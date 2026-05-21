import pytest
from common.utils import clear_repo


def test_clear_repo_delegates_to_clear_vllm_source_tree(mocker):
    mock = mocker.patch("common.utils.clear_vllm_source_tree")
    clear_repo("/some/repo/path")
    mock.assert_called_once_with("/some/repo/path")


def test_clear_repo_raises_for_nonexistent_path():
    with pytest.raises(FileNotFoundError):
        clear_repo("/nonexistent/path/abc123")


def test_clear_repo_raises_for_non_git_repo(tmp_path):
    with pytest.raises(RuntimeError, match="Not a git repository"):
        clear_repo(str(tmp_path))
