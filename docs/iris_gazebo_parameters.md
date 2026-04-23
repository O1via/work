# Gazebo 官方 Iris 参数表（PX4-SITL gazebo-classic）

数据来源：
- https://raw.githubusercontent.com/PX4/PX4-SITL_gazebo-classic/main/models/iris/iris.sdf.jinja

> 说明：下表按控制建模最常用字段整理，字段名与 SDF 保持一致。

## 1) 机体与惯性参数

| 类别 | SDF字段 | 中文名称 | 数值 | 单位 | 备注 |
|---|---|---:|---:|---|---|
| 机体 | mass | 机体质量 | 1.5 | kg | base_link 惯量块 |
| 惯量 | ixx | x轴转动惯量 | 0.029125 | kg·m² | base_link |
| 惯量 | iyy | y轴转动惯量 | 0.029125 | kg·m² | base_link |
| 惯量 | izz | z轴转动惯量 | 0.055225 | kg·m² | base_link |

## 2) 旋翼几何位置（相对 base_link）

| 旋翼 | pose(x,y,z) | 单位 | 备注 |
|---|---:|---|---|
| rotor_0 | (0.13, -0.22, 0.023) | m | 前右 |
| rotor_1 | (-0.13, 0.2, 0.023) | m | 后左 |
| rotor_2 | (0.13, 0.22, 0.023) | m | 前左 |
| rotor_3 | (-0.13, -0.2, 0.023) | m | 后右 |

## 3) 电机/桨盘模型参数（4个旋翼一致）

| SDF字段 | 中文名称 | 数值 | 常见符号 | 单位 | 备注 |
|---|---|---:|---|---|---|
| timeConstantUp | 电机升速时间常数 | 0.0125 | - | s | 电机一阶上升动态 |
| timeConstantDown | 电机降速时间常数 | 0.025 | - | s | 电机一阶下降动态 |
| maxRotVelocity | 最大旋转角速度 | 1100 | $\omega_{max}$ | rad/s | 单电机上限 |
| motorConstant | 推力系数 | 5.84e-06 | $k_f$ | N·s²/rad² | 常见关系 $T=k_f\omega^2$ |
| momentConstant | 反扭矩系数 | 0.06 | $k_m$ | - | 偏航反扭矩比例 |
| rotorDragCoefficient | 旋翼阻力系数 | 0.000175 | - | - | 空气阻力项 |
| rollingMomentCoefficient | 滚转力矩系数 | 1e-06 | - | - | 附加滚转力矩项 |
| rotorVelocitySlowdownSim | 仿真转速缩放系数 | 10 | - | - | Gazebo 数值稳定缩放 |

## 4) 旋翼转向

| 旋翼 | turningDirection | 中文 |
|---|---|---|
| rotor_0 | ccw | 逆时针 |
| rotor_1 | ccw | 逆时针 |
| rotor_2 | cw | 顺时针 |
| rotor_3 | cw | 顺时针 |

## 5) 与当前代码直接关联的核心参数

- 质量：$m=1.5$ kg
- 推力系数：$k_f=5.84\times10^{-6}$
- 最大转速：$\omega_{max}=1100$ rad/s

可得到总推力上限估算：

$$
T_{max}=4k_f\omega_{max}^2\approx 28.2656\ \text{N}
$$

若悬停推力 $T_{hover}=mg=1.5\times9.81=14.715\ \text{N}$，则余量约：

$$
\Delta T_{max}=T_{max}-T_{hover}\approx 13.5506\ \text{N}
$$
