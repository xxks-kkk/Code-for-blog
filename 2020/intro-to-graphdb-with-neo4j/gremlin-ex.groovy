/* Find co-stars of The Office
   Tested on gremlin console 3.4.8
   Usage: inside the gremlin console, do
   gremlin> :l gremlin-ex.groovy
*/

graph = TinkerGraph.open()
g = graph.traversal()
v1 = g.addV("Person").property("name", "Steve Carell").next()
v2 = g.addV("Person").property("name", "B.J. Novak").next()
v3 = g.addV("TVSHOW").property("title", "The Office").next()
g.addE("acts_in").from(v1).to(v3)
g.addE("acts_in").from(v2).to(v3)
g.V().has("TVSHOW", "title", "The Office").inE("acts_in").count()
g.V().has("TVSHOW", "title", "The Office").
in('acts_in').hasLabel("Person").
values("name")