# Coding challenge
https://github.com/Isentia/Coding-Challenge.   
This page consists a coding challenge for Data Engineering roles at Isentia.

# Purpose
Aim of this test is three fold,

- evaluate your coding abilities 
- judge your technical experince
- understand how you design a solution

# How you will be judged
You will be scored on,

- coding standard, comments and style
- unit testing strategy
- overall solution design
- appropriate use of source control

# Intructions

- Please use hosted MongoDB at [Compose](https://compose.io/) for storing the results of the scraped documents
- Candidate should put their test results on a public code repository hosted on Github
- Once test is completed please share the Github repository URL to hiring team so they can review your work
- You are building a backend application and no UI is required, input can be provided using a configuration file or command line

# Challenge - News Content Collect and Store

Create a solution that crawls for articles from a news website, cleanses the response, stores in a mongo database then makes it available to search via an API.

## Details

- Write an application to crawl an online news website, e.g. www.theguardian.com/au or www.bbc.com using a crawler framework such as [Scrapy] (http://scrapy.org/). You can use a crawl framework of your choice and build the application in Python.
- The appliction should cleanse the articles to obtain only information relevant to the news story, e.g. article text, author, headline, article url, etc.  Use a framework such as Readability to cleanse the page of superfluous content such as advertising and html
- Store the data in a hosted mongo database, e.g. compose.io/mongo, for subsequent search and retrieval.  Ensure the URL of the article is included to enable comparison to the original.
- Write an API that provides access to the content in the mongo database.  The user should be able to search for articles by keyword

# Bonus point
If you deploy the solution as a public API using an Amazon EC2 instance.
