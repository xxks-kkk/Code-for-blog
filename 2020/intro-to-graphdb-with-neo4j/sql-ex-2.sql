drop table if exists dunderMifflin;

create table dunderMifflin (
       empID int PRIMARY KEY,
       name char(30),
       role text[],                       
       dept text[],
       mgrID int,
       CONSTRAINT employees_manager_fk FOREIGN KEY (mgrID) REFERENCES dunderMifflin(empID) on delete set null on update cascade);

insert into dunderMifflin (empID, name, role, dept, mgrID) values
(1, 'David Wallace', '{"CFO"}', '{"management"}', null),
(2, 'Ryan Howard', '{"VP, North East Region"}', '{"management"}', 1),
(3, 'Tobby Flenderson', '{"HR Rep."}', '{"HR"}', 2),
(4, 'Michael Scott', '{"Regional Manager"}', '{"management"}', 2),
(5, 'Todd Pecker', '{"Travel Sales Rep."}', '{"Sales"}', 2),
(6, 'Angela Martin', '{"Senior Accountant"}', '{"Accounting", "Party Planning Committee"}', 4),
(7, 'Dwight Schrute', '{"Sales", "Assistant to the Regional Manager"}', '{"Sales"}', 4),
(8, 'Jim Halpert', '{"Sales", "Assistant Regional Manager"}', '{"Sales"}', 4),
(9, 'Pam Beesley', '{"Receptionist"}','{"Reception", "Party Planning Committee"}', 4),
(10, 'Creed Barton', '{"Quality Assurance Rep."}', '{"Quality Control"}', 4),
(11, 'Darryl Philbin', '{"Warehouse Foreman"}', '{"Warehouse"}', 4),
(12, 'Kevin Malone', '{"Accountant"}', '{"Accounting"}', 6),
(13, 'Oscar Martinez', '{"Accountant"}', '{"Accounting"}', 6),
(14, 'Meredith Palmer', '{"Supplier Relations"}', '{"Supplier Relations", "Party Planning Committee"}', 4),
(15, 'Kelly Kapoor', '{"Customer Service Rep."}', '{"Customer Service", "Party Planning Committee"}', 4),
(16, 'Jerry DiCanio', null, '{"Warehouse"}', 11),
(17, 'Madge Madsen', null, '{"Warehouse"}', 11),
(18, 'Lonnie Collins', null, '{"Warehouse"}', 11),
(19, 'Andy Bernard', '{"Regional Director in Sales"}', '{"Sales"}', 8),
(20, 'Phyllis Lapin', '{"Sales"}', '{"Sales", "Party Planning Committee"}', 19),
(21, 'Stanley Hudson', '{"Sales"}', '{"Sales"}', 19)
;

with recursive samelevel(s1, s2, s3, s4) as (
     (select a1.name, a1.mgrID, a2.name, a2.mgrID
      from dunderMifflin a1, dunderMifflin a2
      where a1.mgrID = a2.mgrID)
     union
     (select a1.name, a1.mgrID, a2.name, a2.mgrID
      from dunderMifflin a1, dunderMifflin a2, samelevel l1
      where a1.mgrID = l1.s2 and a2.mgrID = l1.s4)
) select l2.s3 from samelevel L2 where l2.s1 = 'Michael Scott' and l2.s1 <> l2.s3;
