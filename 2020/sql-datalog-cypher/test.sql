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

