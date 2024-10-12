# FRA333_HW3_6521_6567 

จัดทำโดย 
- ณัชณศา_6521
- ชัญญาภัค_6567

File 
- Answer : FRA333_HW3_6521_6567.py
- Check Answer : testScript.ipynb 

### The concept of checking answers and the results of checking

- Find DH Parameter of Robot 3DOF
![image](https://github.com/user-attachments/assets/1c3dbe45-c017-46f6-868f-a328535bc8ce)

Import Library 
```bash
import roboticstoolbox as rtb
import numpy as np

from spatialmath import SE3
from math import pi
from HW3_utils import *
from FRA333_HW3_6521_6567 import *
```

Parameter of Robot 
```bash
d1 = 0.0892
a2 = -0.425
a3 = -0.39243
d4 = 0.109
d5 = 0.093
d6 = 0.082
```

DH Parameter (Modify)
```bash
T3_e = SE3(a3-d6,-d5,d4) * SE3.RPY(0,-pi/2,0)

robot = rtb.DHRobot(
    [
        rtb.RevoluteMDH(d= d1 ,offset= pi), 
        rtb.RevoluteMDH(alpha= pi/2), 
        rtb.RevoluteMDH(a= a2),
    ]
    ,tool = T3_e, name="3R robot")

print(robot)
```

- Result

![image](https://github.com/user-attachments/assets/503c737d-aa4d-4c0a-9922-12eb3d5e18cb)

## Prove1 : Jacobian

### solution 
```bash
def endEffectorJacobianHW3(q:list[float])->list[float]:

    #Get Rotation Matrix and Traslation Matrix from HW3_utils.py
    R,P,R_e,p_e = HW3_utils.FKHW3(q)

    #Create Jacobian Matrix (6 row and 3 column)
    J_q = np.empty((6,3))

    for i in range(len(q)):

        #Jacobian linear velocity 
        J_v = np.cross(R[:,:,i][:,2],(p_e - P[:,i])) 

        #Jacobian angular velocity
        J_w = R[:,:,i][:,2]     

        #Join matrix of linear velocity and angular velocity                          
        J_i = np.concatenate((J_v,J_w),axis=0)       

        #Append Jacobian of each joint
        J_q[:,i] = J_i   

    return J_q
```

### check answer  
```bash
def proofJacobian(q: list[float], robot: rtb.DHRobot) -> bool:
    # Jacobian from my answer 
    J_e = endEffectorJacobianHW3(q) 

    # Jacobian from Robotics Toolbox (reference base frame)
    J_ertb = robot.jacob0(q) 

    # Tolerance as needed
    allow_error = 0.0001  
    
    print(q)
    print("-----------Jacobian จากโค้ด-----------")
    print(J_e)
    print("-----------Jacob0 จาก Robotics Toolbox------------")
    print(J_ertb)
    print("------------Jacobian ถูกหรือไม่------------")
    
    # Compare Jacobian (my answer VS Robotics Toolbox)
    return np.allclose(J_e, J_ertb, atol=allow_error)

# Random q 
q = np.random.rand(3) * 2 * np.pi

result = proofJacobian(q, robot)

print("Jacobian is correct:", result)
```

- Result

![image](https://github.com/user-attachments/assets/bb6fcfc2-f38c-4727-a7c8-ebfc1b9ce141)

- เปรียบเทียบ Jacobian ที่หาจาก solution VS Jacobian ที่หาจาก Robotic Toolbox 
  - Jacobian ที่หาจาก solution : สร้าง jacobian matrix ขนาด 6x3 โดย 3 แถวแรกเป็น linear velocity และ 3 แถวสุดท้ายเป็น angular velocity ในรูปแบบ Revolute โดยมี 3 หลักเนื่องจากหุ่นยนต์มี 3 joint 
  - Jacobian ที่หาจาก Robotic Toolbox : คิด DH Parameter ของหุ่นยนต์ และใช้เครื่องมือ Robotic Toolbox ในการหา Jacobian เทียบเฟรม base
- random ค่า q และแทนค่า เพื่อเปรียบเทียบคำตอบระหว่างสองวิธีนี้โดยกำหนดค่าerror ไม่เกิน 0.0001 

## Prove2 : Singularity 

### solution 
```bash
def checkSingularityHW3(q:list[float])->bool:

    J_q = endEffectorJacobianHW3(q)

    #Part of linear velocity(3x3 matrix)
    J_v_part = J_q[:3, :]

    #Determinant of Jacobian
    determinant = np.linalg.det(J_v_part)

    if abs(determinant) < 1e-3:
        #Near Singularity 
        return True
    else:
        #Non Singularity
        return False
```

### check answer  
```bash
def proveSingularity(q: list[float], robot: rtb.DHRobot) -> bool:
    
    is_singular_HW3 = checkSingularityHW3(q)

    # Jacobian from Robotics Toolbox (Reference from base frame)
    J_ertb = robot.jacob0(q)  
    
    # part of linear velocity (3x3 matrix)
    J_v_part = J_ertb[:3, :] 
    
    # Determinant of Jacobian
    determinant = np.linalg.det(J_v_part)
    
    # Check Singularity from determinant
    is_singular_rtb = abs(determinant) < 1e-3

    print("q ที่สุ่มได้:", q)
    print("ค่า Determinant จาก RTB:", determinant)
    print("อยู่ในจุด Singularities จาก RTB:", is_singular_rtb)
    print("อยู่ในจุด Singularities จาก HW3:", is_singular_HW3)

    # Compare the results of the two functions  
    result = is_singular_HW3 == is_singular_rtb
    print("ผลลัพธ์ตรงกันหรือไม่:", result)
    
    return result

# random q for robot 3DOF
q = np.random.rand(3) * 2 * np.pi
print("ผลลัพธ์สุดท้าย:", proveSingularity(q,robot))

# example that Singularity 
# q = [5.88029104 1.70964541 3.03725032]
```

- Result

![image](https://github.com/user-attachments/assets/446f4113-f265-4702-95a3-524891a63507)

- เปรียบเทียบ Singularity ที่หาจาก solution VS Singularity ที่หาจาก Robotic Toolbox 
  - Singularity ที่หาจาก solution : นำ Jacobian ที่หาได้จากข้อที่ 1 มาลดรูปโดยเลือกเฉพาะ 3 แถวแรกที่เป็นส่วน linear velocity เพราะเป็นส่วนที่สามารถควบคุมได้ แล้วนำมาหา determinant โดยหากมีค่าน้อยกว่า 0.001 หมายถึงเข้าใกล้สภาวะ Singularity 
  - Singularity ที่หาจาก Robotic Toolbox : นำJacobian เทียบเฟรม base ที่ได้จาก Robotic Toolbox มาทำเช่นเดียวกับวิธี Solution
- random ค่า q และแทนค่า เพื่อหาสภาวะ Singularity และนำมาเปรียบเทียบกันว่าถูกต้องหรือไม่ 
- ตัวอย่างค่า q ที่ทำให้เกิด Sigularity ex. q = [5.88029104 1.70964541 3.03725032]
