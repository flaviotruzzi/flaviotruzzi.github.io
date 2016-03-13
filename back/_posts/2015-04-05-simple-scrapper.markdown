---
layout: post
title: Simple Json API Crawl Challenge
date: 2015-04-05 21:00:00
tags: python
---

Last friday I was wandering through [Hacker News](https://news.ycombinator.com/) and a company sent a challenge to find
good candidates for hire. The truth is that I was not looking for a job, but wanted the challenge. In fact, the challenge
is not really new, it consisted of a json api that has new nodes for exploring, and you need to discover its secrets.
Unfortunately, the solution for these kinds of crawl based is always the same, nevertheless I need something to write about.

## First things first

When you open the challenge website on the browser, you get:

`On the right track. You can start here: "/start"`

So, you hit /start, and you end up with a plain text:

`
{
  "error": "\"Session\" header is missing. \"/get-session\" to get a session id."
}
`

Looking at the the response with `curl -v` we could see that the Content-Type was set to `application/json` and the 
response code was 404, maybe another response code would be more suitable... When we access the `get-session`, we receive
some kind of hash.

Then we try again in the `/start` endpoint passing Session header with the contents of the get-session response, and our
response finally look like something we could crawl:

```json
{
  "depth": 0, 
  "id": "start", 
  "message": "There is something we want to tell you, let's see if you can figure this out by finding all of our secrets", 
  "next": [
    "34ffe00db65f4576b5add43dda39ff99", 
    "ebdf4d2f11514626a1b07d745d4a0fc6", 
    "64bbc0003e824075ad59fb5cfaaac4cd"
  ]
}
```

## 1. Quick and Dirty

After playing with curl I noticed that the session id was only valid for 10 requests, and then you would need to get 
another one. So I thought ok, lets do something quick and dirty, and then I failed for some minutes, until I saw all the
problems that my code was stumbling.

First, the `next` value sometimes was a list sometimes a string, and then sometimes the keys had some letters in 
uppercase. Then I got something like the following, I omitted the challenge website.

```python
# Simplified API:
# /get-session -> return a json object with {"session-id": "some_hash_string"}
# /hash_id -> if your session is valid:
#                return a json object with {"next": ["hash_1", ..., "hash_n"]},
#                and optionally "secret" with a letter
#             else: {"error": "Get a new session"}
import requests

BASE_URL = "http://###.###.com/"

visited = set()
secret = ""


def get_session():
    return requests.get(BASE_URL + "/get-session").content


def good_object(obj):   
    for key, value in obj.items():
        obj[key.lower()] = value
    return obj


def get_secret(content):
    return content.get('secret', '')


def get_next(next):
    if isinstance(next, list):
        for node_id in next:
            yield BASE_URL + node_id
    else:
        yield BASE_URL + next


def process_node(url, session):
    global visited, secret
    content = good_object(requests.get(url, headers={'Session': session}).json())
    if "error" in content:
        content = good_object(requests.get(url, headers={'Session': get_session()}).json())
    s = get_secret(content)
    secret += s
    for node in get_next(content.get('next', [])):
        if node not in visited:
            visited.add(node)
            process_node(node, session)


process_node(BASE_URL + "start", get_session())
print("Secret {}".format(secret))
```

That's all, I don't have the time for going over the quick and dirty solution :p