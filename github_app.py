from github import Github
from github import Auth
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from phi.agent import Agent
from phi.tools.github import GithubTools
from ghh import GithubTools2


# Load environment variables
load_dotenv()

agent = Agent(
    instructions=["Use your tools to get user details and repositories of the user"],
    tools=[GithubTools2()],
)

agent.print_response("Give me details about ggerganov", markdown=True)

# app = FastAPI()

# origins = [
#     "http://localhost:5173",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# def get_github_client():
#     github_token = os.getenv("GITHUB_TOKEN")
#     if not github_token:
#         raise HTTPException(status_code=500, detail="GitHub token not configured")
#     auth = Auth.Token(github_token)
#     return Github(auth=auth)


# @app.get("/api/github/{username}")
# def get_github_user_data(username: str):
#     g = get_github_client()
#     try:
#         user = g.get_user(username)

#         # Get user details
#         user_data = {
#             "user": {
#                 "login": user.login,
#                 "name": user.name,
#                 "bio": user.bio,
#                 "public_repos": user.public_repos,
#                 "followers": user.followers,
#                 "following": user.following,
#                 "location": user.location,
#                 "email": user.email,
#                 "avatar_url": user.avatar_url,
#                 "company": user.company,
#                 "blog": user.blog,
#                 "twitter_username": user.twitter_username,
#                 "created_at": user.created_at.isoformat() if user.created_at else None,
#                 "updated_at": user.updated_at.isoformat() if user.updated_at else None,
#             },
#             "repositories": [],
#             "projects": [],
#         }

#         # Get repositories
#         for repo in user.get_repos():
#             repo_data = {
#                 "name": repo.name,
#                 "description": repo.description,
#                 "stars": repo.stargazers_count,
#                 "forks": repo.forks_count,
#                 "language": repo.language,
#                 "visibility": repo.visibility,
#                 "topics": repo.get_topics(),
#                 "size": repo.size,
#                 "default_branch": repo.default_branch,
#                 "created_at": repo.created_at.isoformat() if repo.created_at else None,
#                 "updated_at": repo.updated_at.isoformat() if repo.updated_at else None,
#                 "url": repo.html_url,
#             }
#             user_data["repositories"].append(repo_data)

#         # Get projects
#         try:
#             for project in user.get_projects():
#                 project_data = {
#                     "name": project.name,
#                     "body": project.body,
#                     "state": project.state,
#                     "created_at": (
#                         project.created_at.isoformat() if project.created_at else None
#                     ),
#                     "updated_at": (
#                         project.updated_at.isoformat() if project.updated_at else None
#                     ),
#                 }
#                 user_data["projects"].append(project_data)
#         except:
#             # Projects might not be accessible for some users
#             user_data["projects"] = []

#         g.close()
#         return user_data

#     except Exception as e:
#         g.close()
#         raise HTTPException(status_code=404, detail=f"Error fetching data: {str(e)}")
