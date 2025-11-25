Tips and Tricks
===============

Here is a collection of things I recommend you to do when you want to write software that is KRROOD complaint,

Best Practices
---------------------------------
- Model your programs with dataclasses always
- Make inheritance intentional. Put the “queryable” base class first in multiple inheritance to align polymorphic behavior with your expectations.
- Prefer explicit mappings when integrating with classes where the design is made for fast and scalable calculations and not for high level APIs.
- Rely on a `mediator pattern <https://www.youtube.com/watch?v=35D5cBosD4c&>`_ for the integration of all the different information in your program.

Writing Queries
---------------
SQLAlchemy is well known in the training data for LLMs. For most queries, you can just copy the generated interface to
your favourite LLM and then tell it to write a query that does what you want. This is a very nice way of doing
text to sql for your domain.
