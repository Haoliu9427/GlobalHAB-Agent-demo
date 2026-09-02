# 现场前向验证数据接口

`field_observations_template.csv` 与 `field_currents_template.csv` 仅用于说明字段格式；其中示例行不是现场实测数据。

最低输入：

- 观测：`date, station_id, latitude, longitude, cell_count`
- 流场：`date, latitude, longitude, u_ms, v_ms`

推荐补充：毒素、SST/水温、盐度、DO、NO3、PO4、SiO4、Chl-a、鱼体/鳃部反应、养殖网箱ID与密度等。

系统先执行质量门控；满足前向验证条件后，只在较早时间块选择候选lag，再在后续时间块一次性评估。若数据不足则返回 `defer`，不会强行输出传播结论。
