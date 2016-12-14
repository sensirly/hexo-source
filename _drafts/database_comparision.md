
# SQL vs NoSQL
- 存储：SQL数据库提供了一族相关的数据表；而NoSQL数据库存储类JSON的键值对文档，更加灵活，可以存储任何形式的数据
- schema less：SQL数据架构一定要在实现应用逻辑之前被设计和实现，修改成本很大；在NoSQL数据库中，数据可以被非常灵活的添加，更适合初始数据形式很难确定的项目中。
- JOIN：SQL查询提供了强劲的JOIN语法。NoSQL没有相对应的JOIN，也是为什么对NoSQL往往使用去中心化的方式很有必要。
- 事务：在SQL数据库中，多条更新语句能够在一个事务中被同时执行；在NoSQL数据库中，对单个文档的修改是原子的。
- 性能：去中心化存储允许你在单次请求中获取一个条目的所有信息，通常比需要多次JOIN的SQL速度快。


# 关系数据库

# 内存数据库(In-memory database,IMDB,MMDB)

# key-value 数据库

# 图数据库
图数据库由点和边组成。点代表要表示的实体；边表示实体间的关系，均为有向边。点和边都有多个属性和标签。以Neo4j为代表    
![](/img/system_design/graph_database.png)    

## 属性的存储
节点属性既可以采用Key-Value存储表独立存储，也可以冗余存放在关系表的Value中。两者优缺点互补，采用独立存储：
- 可以节省存储空间，每个节点只需要保存一份属性
- 更新方便，每个属性只需要更新一次
- 可以直接根据节点id访问属性
- 影响访问性能，在访问关系数据的同时需要访问节点属性数据，那么需要通过一次额外请求

## 和Relationship Database比较   
![](/img/system_design/neo4jvsrdbms.png)   
关系型数据库适合扁平的数据结构，数据间的关系只有一两层，仅用很少的join操作就能完成所有的查询。图数据库适合包含很多链接的数据，比如社交网络关系。适合大量在线查询的场景。

# Giraph
Giraph是一个分布式平台，利用了Hadoop的部分实现原理，但是只保留了Map部分，并且存储是基于内存的，但是数据量太大是会有存储困难。Neo4j是单机的存储平台，通过两层内存缓存机制提高处理性能，使用双向链表存储数据，便于对图进行遍历完成计算。

[Neo4j vs giraph - presentations](http://www.slideshare.net/nishantgandhi99/neo4j-vs-giraph)