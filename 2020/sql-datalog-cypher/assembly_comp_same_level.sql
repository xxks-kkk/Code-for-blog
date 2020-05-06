DROP TABLE IF EXISTS AssemblyInstance;

CREATE TABLE AssemblyInstance(
       PART    VARCHAR(10),
       SUBPART VARCHAR(10),
       NUMBER  INT);

INSERT INTO AssemblyInstance(PART, SUBPART, NUMBER) VALUES
       ('trike', 'wheel' , 3),
       ('trike', 'frame', 1),
       ('frame', 'seat', 1),
       ('frame', 'pedal', 2),
       ('wheel', 'spoke', 2),
       ('wheel', 'tire', 1),
       ('tire', 'rim', 1),
       ('tire', 'tube', 1);

-- https://www.postgresql.org/docs/12/queries-with.html
WITH RECURSIVE Comp(PART, SUBPART) AS (
       (SELECT A1.PART, A1.SUBPART FROM AssemblyInstance A1)
     UNION
       (SELECT A2.PART, C1.SUBPART
        FROM AssemblyInstance A2, Comp C1
        WHERE A2.SUBPART = C1.PART)
)
SELECT * FROM Comp C2 WHERE C2.PART = 'trike';

WITH RECURSIVE SameLev(S1, S2) AS (
     (SELECT A1.SUBPART, A2.SUBPART
      FROM AssemblyInstance A1, AssemblyInstance A2
      WHERE A1.PART = A2.PART)
     UNION
     (SELECT A1.SUBPART, A2.SUBPART
     FROM AssemblyInstance A1, AssemblyInstance A2, SameLev L1
     WHERE L1.S1 = A1.PART AND L1.S2 = A2.PART)
)
SELECT * FROM SameLev L2 WHERE L2.S1 = 'spoke' AND L2.S1 <> L2.S2;
     
