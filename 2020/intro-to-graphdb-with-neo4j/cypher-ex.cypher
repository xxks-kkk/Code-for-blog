CREATE
(n1:Person {name: "Steve Carell", gender: "male"}),
(n2:Person {name: "B.J. Novak", gender: "male"}),
(n3:TVSHOW {title: "The Office"}),
(n1)-[:acts_in {role: "Michael G. Scott", ref: "Wikipedia"}]->(n3),
(n2)-[:acts_in {role: "Ryan Howard", ref: "Wikipedia"}]->(n3),
(n2)-[:produces]->(n3);

CREATE
(n1:Person {name: "David Wallace", role: "CFO", dept: "management"}),
(n2:Person {name: "Ryan Howard", role: "VP, North East Region", dept: "management"}),
(n3:Person {name: "Toby Flenderson", role: "HR Rep.", dept: "HR"}),
(n4:Person {name: "Michael Scott", role: "Regional Manager", dept: "management"}),
(n5:Person {name: "Todd Pecker", role: "Travel Sales Rep.", dept: "Sales"}),
(n6:Person {name: "Angela Martin", role: "Senior Accountant", dept: ["Accounting", "Party Planning Committee"]}),
(n7:Person {name: "Dwight Schrute", role: ["Sales", "Assistant to the Regional Manager"], dept: "Sales"}),
(n8:Person {name: "Jim Halpert", role: ["Sales", "Assistant Regional Manager"], dept: "Sales"}),
(n9:Person {name: "Pam Beesley", role: "Receptionist", dept: ["Reception", "Party Planning Committee"]}),
(n10:Person {name: "Creed Barton", role: "Quality Assurance Rep.", dept: "Quality Control"}),
(n11:Person {name: "Darryl Philbin", role: "Warehouse Foreman", dept: "Warehouse"}),
(n12:Person {name: "Kevin Malone", role: "Accountant", dept: "Accounting"}),
(n13:Person {name: "Oscar Martinez", role: "Accountant", dept: "Accounting"}),
(n14:Person {name: "Meredith Palmer", role: "Supplier Relations", dept: ["Supplier Relations", "Party Planning Committee"]}),
(n15:Person {name: "Kelly Kapoor", role: "Customer Service Rep.", dept: ["Customer Service", "Party Planning Committee"]}),
(n16:Person {name: "Jerry DiCanio", dept: "Warehouse"}),
(n17:Person {name: "Madge Madsen", dept: "Warehouse"}),
(n18:Person {name: "Lonnie Collins", dept: "Warehouse"}),
(n19:Person {name: "Andy Bernard", role: "Regional Director in Sales", dept: "Sales"}),
(n20:Person {name: "Phyllis Lapin", role: "Sales", dept: ["Sales", "Party Planning Committee"]}),
(n21:Person {name: "Stanley Hudson", role: "Sales", dept: "Sales"}),
(n1)-[:manages]->(n2),
(n2)-[:manages]->(n3),
(n2)-[:manages]->(n4),
(n2)-[:manages]->(n5),
(n4)-[:manages]->(n6),
(n4)-[:manages]->(n7),
(n4)-[:manages]->(n8),
(n4)-[:manages]->(n9),
(n4)-[:manages]->(n10),
(n4)-[:manages]->(n11),
(n4)-[:manages]->(n14),
(n4)-[:manages]->(n15),
(n6)-[:manages]->(n12),
(n6)-[:manages]->(n13),
(n8)-[:manages]->(n19),
(n11)-[:manages]->(n16),
(n11)-[:manages]->(n17),
(n11)-[:manages]->(n18),
(n19)-[:manages]->(n20),
(n19)-[:manages]->(n21);

MATCH (p:Person)
WHERE "Party Planning Committee" in p.dept
return p.name

MATCH p1 = (p:Person)<-[:manages]-(n:Person)
WHERE n.name = "Michael Scott"
RETURN count(p1);

MATCH p1 = (n:Person)<-[:manages]-(p:Person)
MATCH p2 = (m:Person)<-[:manages]-(p:Person)
WHERE length(p1) = length(p2) AND m.name <> n.name AND n.name = "Michael Scott"
RETURN m;

MATCH (p {name: 'Michael Scott'})-[:manages]->()-[:manages]->(fof)
RETURN fof.name;

MATCH (p:Person)<-[:manages]-(n:Person)
WHERE n.name = "Michael Scott"
WITH count(p) AS c1
MATCH (p:Person)<-[:manages]-(m:Person)
WHERE m.name = "Jim Halpert"
RETURN c1 > count(p)

MATCH path=(p:Person)-[:manages*1..]->(q:Person)
WHERE p.name = "Michael Scott"
return q.name

MATCH path=(p:Person)-[:manages*1..]->(q:Person)
WHERE p.name = "Jim Halpert" and q.name = "Phyllis Lapin"
return count(path) > 0

MATCH path=(p1:Person {name: "Michael Scott"})-[:manages*1..]->()-[:manages*1..]->(p2:Person)
return collect(distinct p2)

MATCH path = shortestPath((p:Person {name: "David Wallace"})-[:manages*1..]-(q:Person {name: "Andy Bernard"}))
RETURN path
