---
layout: post
title: Simple Json API
date: 2015-04-05 21:00:00
tags: python
---

Last friday I was wandering through [Hacker News](https://news.ycombinator.com/) and a company sent a challenge to find
good candidates for hire. The truth is that I am not looking for a job, but wanted the challenge. In fact, the challenge
is not really new, it consisted of a json api that has new nodes for exploring, and you need to discover its secrets.
The solution was straight forward: [depth-first search](http://en.wikipedia.org/wiki/Depth-first_search).

The most interesting thing, and in fact what make the challenge kinda challenging was two things:

1. The api used an expiring session, you need to get a session and you could use that session-id only for 10 requests,
after that you need to get another session-id,
2. The api was messed up. Yeah, the response mixed lower and upper case.

## 1. Handling the first issue

The first issue we need to solve was the session-id behaviour. The first thing one would come up with is:

```python
# Simplified API:
# /get-session -> return a json object with {"session-id": "some_hash_string"}
# /hash_id -> if your session is valid:
#                return a json object with {"next": ["hash_1", ..., "hash_n"]},
#                and optionally "secret" with a letter
#             else: {"error": "Get a new session"}
import requests

secret = ""
visited = set()

def get_session():
    response = requests.get("/get-session").json()
    return response["session-id"]

def transverse_nodes(hash, session_id):
    response = requests.get(hash, headers={"Session": session_id}).json()
    if "error" in response:
        session_id = get_session()
        response = requests.get(hash, headers={"Session": session_id}).json()
    if "secret" in response:
        secret += response["secret"]
    for new_hash in response["next"]:
        if new_hash not in visited:
            visited.add(new_hash)
            transverse_nodes(new_hash, session_id)

transverse_nodes("initial_hash", get_session())

print secret
```
