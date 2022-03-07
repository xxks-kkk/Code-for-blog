// Ideally, to create a graph that contains one triangle,
// I want to do

// CREATE
// (v1:vertex),
// (v2:vertex),
// (v3:vertex),
// (v4:vertex),
// (v5:vertex),
// (v6:vertex),
// (v7:vertex),
// (v8:vertex),
// (v9:vertex);

// CREATE (v4)-[:e]->(v1);
// CREATE (v4)-[:e]->(v2);
// CREATE (v4)-[:e]->(v3);
// CREATE (v4)-[:e]->(v5);
// CREATE (v4)-[:e]->(v6);
// CREATE (v6)-[:e]->(v8);
// CREATE (v6)-[:e]->(v9);
// CREATE (v6)-[:e]->(v5);
// CREATE (v6)-[:e]->(v7);
// CREATE (v7)-[:e]->(v4);

// However, I end up doing the following (clearly, label itself just doesn't help
// to uniquely identify a node)

CREATE
(v1:vertex {name: 'v1'}),
(v2:vertex {name: 'v2'}),
(v3:vertex {name: 'v3'}),
(v4:vertex {name: 'v4'}),
(v5:vertex {name: 'v5'}),
(v6:vertex {name: 'v6'}),
(v7:vertex {name: 'v7'}),
(v8:vertex {name: 'v8'}),
(v9:vertex {name: 'v9'});

MATCH (v4:vertex),(v1:vertex)
WHERE v4.name = 'v4' and v1.name = 'v1'
CREATE (v4)-[:e]->(v1);

MATCH (v4:vertex),(v2:vertex)
WHERE v4.name = 'v4' and v2.name = 'v2'
CREATE (v4)-[:e]->(v2);
       
MATCH (v4:vertex),(v3:vertex)
WHERE v4.name = 'v4' and v3.name = 'v3'
CREATE (v4)-[:e]->(v3);
       
MATCH (v4:vertex),(v5:vertex)
WHERE v4.name = 'v4' and v5.name = 'v5'
CREATE (v4)-[:e]->(v5);
       
MATCH (v4:vertex),(v6:vertex)
WHERE v4.name = 'v4' and v6.name = 'v6'
CREATE (v4)-[:e]->(v6);
       
MATCH (v6:vertex),(v8:vertex)
WHERE v6.name = 'v6' and v8.name = 'v8'
CREATE (v6)-[:e]->(v8);
       
MATCH (v6:vertex),(v9:vertex)
WHERE v6.name = 'v6' and v9.name = 'v9'
CREATE (v6)-[:e]->(v9);
       
MATCH (v6:vertex),(v5:vertex)
WHERE v6.name = 'v6' and v5.name = 'v5'
CREATE (v6)-[:e]->(v5);
       
MATCH (v6:vertex),(v7:vertex)
WHERE v6.name = 'v6' and v7.name = 'v7'
CREATE (v6)-[:e]->(v7);
       
MATCH (v7:vertex),(v4:vertex)
WHERE v7.name = 'v7' and v4.name = 'v4'
CREATE (v7)-[:e]->(v4);

match (x)-[]->(y)-[]->(z)-[]->(x)
return x,y,z;
