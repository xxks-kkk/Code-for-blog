## SQL v. Datalog v. Cypher

In this document, we use a simple table shown below to compare
SQL, Datalog, and Cypher queries.

The working table

| Part  | Subpart | number |
|-------|---------|--------|
| trike | wheel   |  3     |
| trike | frame   | 1      |
| frame | seat    | 1      |
| frame | pedal   | 2      |
| wheel | spoke   | 2      |
| wheel | tire    | 1      |
| tire  | rim     | 1      |
| tire  | tube    | 1      |

## SQL

We use PostgreSQL 12 for our experiment. 


## Datalog

We use [DrRacket](https://docs.racket-lang.org/datalog/Tutorial.html?q=datalog) for this section.


## Cypher
