import os
import json
from typing import Optional, List

from phi.tools import Toolkit
from phi.utils.log import logger

try:
    from github import Github, GithubException, Auth
except ImportError:
    raise ImportError(
        "`PyGithub` not installed. Please install using `pip install PyGithub`"
    )


class GithubTools2(Toolkit):
    def __init__(
        self,
        access_token: Optional[str] = None,
        base_url: Optional[str] = None,
        search_repositories: bool = True,
        list_repositories: bool = True,
        get_repository: bool = True,
    ):
        super().__init__(name="github")

        self.access_token = access_token or os.getenv("GITHUB_ACCESS_TOKEN")
        self.base_url = base_url

        self.g = self.authenticate()

        if search_repositories:
            self.register(self.search_repositories)
        if list_repositories:
            self.register(self.list_repositories)
        if get_repository:
            self.register(self.get_repository)

    def authenticate(self):
        """Authenticate with GitHub using the provided access token."""
        auth = Auth.Token(self.access_token)
        if self.base_url:
            logger.debug(f"Authenticating with GitHub Enterprise at {self.base_url}")
            return Github(base_url=self.base_url, auth=auth)
        else:
            logger.debug("Authenticating with public GitHub")
            return Github(auth=auth)

    def search_repositories(
        self, query: str, sort: str = "stars", order: str = "desc", per_page: int = 5
    ) -> str:
        """Search for repositories on GitHub.

        Args:
            query (str): The search query keywords.
            sort (str, optional): The field to sort results by. Can be 'stars', 'forks', or 'updated'. Defaults to 'stars'.
            order (str, optional): The order of results. Can be 'asc' or 'desc'. Defaults to 'desc'.
            per_page (int, optional): Number of results per page. Defaults to 5.

        Returns:
            A JSON-formatted string containing a list of repositories matching the search query.
        """
        logger.debug(f"Searching repositories with query: '{query}'")
        try:
            repositories = self.g.search_repositories(
                query=query, sort=sort, order=order
            )
            repo_list = []
            for repo in repositories[:per_page]:
                repo_info = {
                    "full_name": repo.full_name,
                    "description": repo.description,
                    "url": repo.html_url,
                    "stars": repo.stargazers_count,
                    "forks": repo.forks_count,
                    "language": repo.language,
                }
                repo_list.append(repo_info)
            return json.dumps(repo_list, indent=2)
        except GithubException as e:
            logger.error(f"Error searching repositories: {e}")
            return json.dumps({"error": str(e)})

    def list_repositories(self) -> str:
        """List all repositories for the authenticated user.

        Returns:
            A JSON-formatted string containing a list of repository names.
        """
        logger.debug("Listing repositories")
        try:
            repos = self.g.get_user().get_repos()
            repo_names = [repo.full_name for repo in repos]
            return json.dumps(repo_names, indent=2)
        except GithubException as e:
            logger.error(f"Error listing repositories: {e}")
            return json.dumps({"error": str(e)})

    def get_repository(self, repo_name: str) -> str:
        """Get details of a specific repository.

        Args:
            repo_name (str): The full name of the repository (e.g., 'owner/repo').

        Returns:
            A JSON-formatted string containing repository details.
        """
        logger.debug(f"Getting repository: {repo_name}")
        try:
            repo = self.g.get_repo(repo_name)
            repo_info = {
                "name": repo.full_name,
                "description": repo.description,
                "url": repo.html_url,
                "stars": repo.stargazers_count,
                "forks": repo.forks_count,
                "open_issues": repo.open_issues_count,
                "language": repo.language,
                "license": repo.license.name if repo.license else None,
                "default_branch": repo.default_branch,
            }
            return json.dumps(repo_info, indent=2)
        except GithubException as e:
            logger.error(f"Error getting repository: {e}")
            return json.dumps({"error": str(e)})

    def get_user(self, username: str) -> str:
        """Get details of a specific user.

        Args:
            username (str): The username of the user.

        Returns:
            A JSON-formatted string containing user details.
        """
        logger.debug(f"Getting user: {username}")
        try:
            user = self.g.get_user(username)
            user_info = {
                "name": user.name,
                "username": user.login,
                "url": user.html_url,
                "followers": user.followers,
                "following": user.following,
            }
            if self.list_repositories:
                user_info["repositories"] = [
                    repo.full_name for repo in user.get_repos()
                ]
            return json.dumps(user_info, indent=2)
        except GithubException as e:
            logger.error(f"Error getting user: {e}")
            return json.dumps({"error": str(e)})
