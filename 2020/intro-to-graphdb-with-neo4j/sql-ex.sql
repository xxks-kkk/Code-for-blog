drop table if exists person, tvshow, produces, acts_in;

create table person(pid char(10) primary key, pname char(20), gender char(10));
create table tvshow(tid char(10) primary key, title char(10));
create table produces(id1 char(10), id2 char(10),
    constraint person_id1_fk foreign key (id1) references person(pid),
    constraint title_id2_fk foreign key (id2) references tvshow(tid));
create table acts_in (id1 char(10), id2 char(10), role char(20), ref char(20),
    constraint person_id1_fk foreign key (id1) references person(pid),
    constraint title_id2_fk foreign key (id2) references tvshow(tid));

insert into person(pid, pname, gender) values
('n1', 'Steve Carell', 'male'),
('n3', 'B.J. Novak', 'male');

insert into tvshow(tid, title) values
('n2', 'The Office');

insert into produces(id1, id2) values
('n3', 'n2');

insert into acts_in (id1, id2, role, ref) values
('n3', 'n2', 'Ryan Howard', 'Wikipedia'),
('n1', 'n2', 'Michael G. Scott', 'Wikipedia');

select * from person;
select * from tvshow;
select * from produces;
select * from acts_in;
