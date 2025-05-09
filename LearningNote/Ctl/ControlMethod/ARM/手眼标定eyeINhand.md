# 手眼标定方法（Eye-in-Hand）

## 1. Why 为什么要进行手眼标定

​	已知你有一个机械臂，并且能够成功读取固定在夹具上的相机视图，那么该相机就是指导机械臂进行夹取的眼睛。与人类似，你的眼睛也是你在夹取时的指导。

​	设想这么一个情况，你想移动你的鼠标，但你的眼睛只是一个很普通的RGB-D相机，大多数时候，大脑作为主控，只接收了以你的世界坐标系出发，射向以你的颈部以上的rpy坐标的方向方向上压缩的二维图像。***你想关掉该死的小广告***，想赶紧移动你的鼠标到右上角。

​	事实上，对于你来说，鼠标在你的二维坐标系中从A点移动到了B点，但对于三维世界来说，你将他从a点移动到了b点。

​	之后，才发生了手臂的移动，手掌的握取环节。

​	手眼标定，就是其中的将二维坐标转换为三维坐标的过程。因此，抓取作为机械臂的使命，手眼标定是其中不可或缺的关键步骤之一。

## 2. What 什么是手眼标定

​	前面抽象的讲述了，手眼标定大概是一个***坐标系转换***的过程。对于机械臂来说，他具有这么几个坐标系，夹具坐标系、机械臂baselink的基坐标系、相机坐标系、与被抓取物体（或在标定过程中，我们称之为标定板坐标系）。

​	事实上，具有这么多坐标系，大致也能列出来几个***坐标系转换矩阵***了，从物理角度与工作意义上考虑，转换矩阵分别有四个。

1. **夹具坐标系与基坐标系**的转换矩阵

   这一段应当是已知的（如果你到了这一步，机械臂的控制应当使你能够读取机械臂的关节角度，并计算出最终的转换关系），我们将其记录为***A***。

2. **相机坐标系与夹具坐标系**的转换矩阵

   这一步，由于不同的机械臂，不同的工作目的，相机固定的位置千差万别，我们姑且将其考虑为位置的，记录为***X***

3. 相机坐标系到标定板坐标系

   这就是相机标定后的意义，在于确定如何找到一个能够统一相机与标定板两个坐标系的关键作用，已知，将其记录为***B***

4. 夹具坐标系到标定板坐标系

   这正是我们所求，不是吗，获取了这两个坐标系，距离抓取仅剩一步之遥了，而且只要其余坐标关系不变，这个转换坐标系应当不变。

## 3. How 如何进行手眼标定

​	那么在第二个环节里，我们简单的讲述了已知量与代求量。

​	什么已知呢？坐标都是已知的、A、B是一直的。

​	什么未知呢？X是未知的。

那么其实具有这样的关系，能够实现X的求取
$$
P_{base}=T^{base}_{tool}P_{tool}=T^{base}_{tool}T^{tool}_{camare}P_{camera}
$$
其中，***Px***表示***x***坐标系下的***P***坐标，那么就有：
$$
T^{tool}_{camare}={T^{base}_{tool}}^{-1}P_{base}{P_{camera}}^{-1}
$$
现在，将机械臂的末端移到位置A：

存在这么四个坐标：
$$
P_{base}=T_{toolA}^{base}P_{toolA} \\
P_{toolA}=T_{camera}^{tool}P_{cameraA}\\
P_{cameraA}={T_{cameraA}^{board}}^{-1}P_{board}\\联立后可得：\\
P_{base} = T_{toolA}^{base}T_{camera}^{tool}{T_{cameraA}^{board}}^{-1}P_{board}\\那么对于位置B，则存在：\\
P_{base} = T_{toolB}^{base}T_{camera}^{tool}{T_{cameraB}^{board}}^{-1}P_{board}
$$
其中，标定板到基坐标系的转换关系保持不变，其转换矩阵也相等，设：
$$
T_{toolA}^{base} = A_1,T_{toolB}^{base} = A_2,{T_{cameraA}^{board}} = B_1,{T_{cameraB}^{board}}=B_2
$$
则：
$$
A_1XB_1^{-1}=A_2XB_2^{-1}\\
(A_2^{-1}A_1)AX=XB(B_2^{-1}B_1)
$$
A通过机器人学获取，B通过相机标定获取，这样就能解出来啦。



​	

2024.11.25 written by SunMingchi