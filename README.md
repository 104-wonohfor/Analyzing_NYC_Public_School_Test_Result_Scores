link: [https://colab.research.google.com/drive/1KX_LdQNtDPmazM541i6R-lpsKlASqTdf?usp=sharing](https://colab.research.google.com/drive/1KX_LdQNtDPmazM541i6R-lpsKlASqTdf?usp=sharing)
---
jupyter:
  colab:
    toc_visible: true
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

<div class="cell markdown" id="Xm4-tmDcnzG2">

**Đề bài**

An automobile company has plans to enter new markets with their existing
products (P1, P2, P3, P4 and P5). After intensive market research,
they’ve deduced that the behavior of new market is similar to their
existing market.

Content In their existing market, the sales team has classified all
customers into 4 segments (A, B, C, D ). Then, they performed segmented
outreach and communication for different segment of customers. This
strategy has work exceptionally well for them.

</div>

<div class="cell markdown" id="DWymHSThn3G8">

# **1. Phân tích tập dữ liệu**

</div>

<div class="cell code" execution_count="1" id="vbM6LzPQb_kH">

``` python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
```

</div>

<div class="cell markdown" id="bBEOU7o7oYqz">

## 1.1 Tổng quan về tập dữ liệu *train*

</div>

<div class="cell code" execution_count="2" id="3x-Ka5jscMT-">

``` python
train_dataset = pd.read_csv('Train.csv')
```

</div>

<div class="cell code" execution_count="3"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:206}"
id="szSZ1wicFlR2" outputId="d73d1839-d9ac-47a3-8e5c-9d894c26121f">

``` python
train_dataset.head()
```

<div class="output execute_result" execution_count="3">

           ID  Gender Ever_Married  Age Graduated     Profession  Work_Experience  \
    0  462809    Male           No   22        No     Healthcare              1.0   
    1  462643  Female          Yes   38       Yes       Engineer              NaN   
    2  466315  Female          Yes   67       Yes       Engineer              1.0   
    3  461735    Male          Yes   67       Yes         Lawyer              0.0   
    4  462669  Female          Yes   40       Yes  Entertainment              NaN   

      Spending_Score  Family_Size  Var_1 Segmentation  
    0            Low          4.0  Cat_4            D  
    1        Average          3.0  Cat_4            A  
    2            Low          1.0  Cat_6            B  
    3           High          2.0  Cat_6            B  
    4           High          6.0  Cat_6            A  

</div>

</div>

<div class="cell code" execution_count="4"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="5J0M0wiOdbe5" outputId="33f66e0c-1d2d-44ae-aea2-f364391b0774">

``` python
train_dataset.info()
```

<div class="output stream stdout">

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8068 entries, 0 to 8067
    Data columns (total 11 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   ID               8068 non-null   int64  
     1   Gender           8068 non-null   object 
     2   Ever_Married     7928 non-null   object 
     3   Age              8068 non-null   int64  
     4   Graduated        7990 non-null   object 
     5   Profession       7944 non-null   object 
     6   Work_Experience  7239 non-null   float64
     7   Spending_Score   8068 non-null   object 
     8   Family_Size      7733 non-null   float64
     9   Var_1            7992 non-null   object 
     10  Segmentation     8068 non-null   object 
    dtypes: float64(2), int64(2), object(7)
    memory usage: 693.5+ KB

</div>

</div>

<div class="cell code" execution_count="5"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:300}"
id="xxNu8nwxBhuF" outputId="daff3d47-666a-4c47-e8f4-4f3dac191ffb">

``` python
train_dataset.describe()
```

<div class="output execute_result" execution_count="5">

                      ID          Age  Work_Experience  Family_Size
    count    8068.000000  8068.000000      7239.000000  7733.000000
    mean   463479.214551    43.466906         2.641663     2.850123
    std      2595.381232    16.711696         3.406763     1.531413
    min    458982.000000    18.000000         0.000000     1.000000
    25%    461240.750000    30.000000         0.000000     2.000000
    50%    463472.500000    40.000000         1.000000     3.000000
    75%    465744.250000    53.000000         4.000000     4.000000
    max    467974.000000    89.000000        14.000000     9.000000

</div>

</div>

<div class="cell code" execution_count="6"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="sSKTiBT2E0EQ" outputId="3d3b1ced-002b-4d69-987a-c2e7332e1ec7">

``` python
# Số giá trị null của tập dữ liệu trên từng cột
train_dataset.isnull().sum()
```

<div class="output execute_result" execution_count="6">

    ID                   0
    Gender               0
    Ever_Married       140
    Age                  0
    Graduated           78
    Profession         124
    Work_Experience    829
    Spending_Score       0
    Family_Size        335
    Var_1               76
    Segmentation         0
    dtype: int64

</div>

</div>

<div class="cell code" execution_count="7"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="O-DR7IatItk0" outputId="8a1fa39c-a4f7-4f68-fb62-e9b6544ce15a">

``` python
# Trích xuất giá trị có trong mỗi cột
column_names = train_dataset.columns
for i in column_names[1:]:
  if i == 0:
    continue
  print(train_dataset[i].unique())
```

<div class="output stream stdout">

    ['Male' 'Female']
    ['No' 'Yes' nan]
    [22 38 67 40 56 32 33 61 55 26 19 70 58 41 31 79 49 18 36 35 45 42 83 27
     28 47 29 57 76 25 72 48 74 59 39 51 30 63 52 60 68 86 50 43 80 37 46 69
     78 71 82 23 20 85 21 53 62 75 65 89 66 73 77 87 84 81 88]
    ['No' 'Yes' nan]
    ['Healthcare' 'Engineer' 'Lawyer' 'Entertainment' 'Artist' 'Executive'
     'Doctor' 'Homemaker' 'Marketing' nan]
    [ 1. nan  0.  4.  9. 12.  3. 13.  5.  8. 14.  7.  2.  6. 10. 11.]
    ['Low' 'Average' 'High']
    [ 4.  3.  1.  2.  6. nan  5.  8.  7.  9.]
    ['Cat_4' 'Cat_6' 'Cat_7' 'Cat_3' 'Cat_1' 'Cat_2' nan 'Cat_5']
    ['D' 'A' 'B' 'C']

</div>

</div>

<div class="cell markdown" id="51jaEncnpE44">

## 1.2. Exploratory Data Analysis + Fill null values + Turn label in to numerical categories

</div>

<div class="cell code" execution_count="8" id="smax2KmlL8tE">

``` python
# Tạo một tập dữ liệu copy từ train_dataset để tránh thay đổi dữ liệu gốc
train = train_dataset.copy()
```

</div>

<div class="cell markdown" id="Zw-UwlJ6pZp9">

### 1.2.1 Cột "Gender"

</div>

<div class="cell code" execution_count="9"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="4QOqH9V5MEfk" outputId="d1e5f43e-7f9b-4090-fa23-7a569bc9937b">

``` python
# Số lượng các giá trị trong cột "Gender" theo từng segment
counts = train.groupby(['Gender', 'Segmentation']).size()
print(counts)
```

<div class="output stream stdout">

    Gender  Segmentation
    Female  A                909
            B                861
            C                922
            D                959
    Male    A               1063
            B                997
            C               1048
            D               1309
    dtype: int64

</div>

</div>

<div class="cell code" execution_count="10"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:572}"
id="nWUiwhjzMKsK" outputId="741bed15-e135-452d-e551-0e8c4d50ff4c">

``` python
# Visualization
counts_df = counts.unstack(level=0)
counts_df.plot(kind='bar')

# Thêm tiêu đề và các nhãn cho biểu đồ
plt.xlabel('Segment')
plt.ylabel('Counts')
plt.legend(['Female', 'Male'])
plt.title(str(counts_df))
plt.show()
```

<div class="output display_data">

![](f4a563b7fcf74da691aab075ea23597c868dc818.png)

</div>

</div>

<div class="cell markdown" id="T96jBhTAYjUt">

*Nhận xét: - Trong tất cả các segment, "Female" luôn nhiều hơn "Male"*

</div>

<div class="cell code" execution_count="11" id="rQH3RPjUFOfy">

``` python
# Turn label in to numerical categories
train.Gender = pd.Categorical(train.Gender,categories=['Male','Female'],ordered=True).codes
```

</div>

<div class="cell markdown" id="ky8pLFRepxAc">

### 1.2.2. Cột "Ever_Married"

</div>

<div class="cell code" execution_count="12"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="rw06ocXLJ7Zi" outputId="dd0b9496-8555-450e-9b65-f2536f9ab255">

``` python
# Các giá trị trong cột 'Ever_Married'
print("null values:", train.Ever_Married.isnull().sum())
print(train.Ever_Married.value_counts())
```

<div class="output stream stdout">

    null values: 140
    Yes    4643
    No     3285
    Name: Ever_Married, dtype: int64

</div>

</div>

<div class="cell code" execution_count="13"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:573}"
id="FHNCvuhmVMCe" outputId="022051f8-e945-4fd2-e0eb-aae975f615a9">

``` python
# Visualization
counts = train.groupby(['Ever_Married', 'Segmentation']).size()
counts_df = counts.unstack(level=0)
counts_df.plot(kind='bar')

# Thêm nhãn vầ title
plt.xlabel('Segment')
plt.ylabel('Counts')
plt.title(str(counts_df))
plt.show()
```

<div class="output display_data">

![](3030ed5e80d23619cbcbfbf9c298dba30a174bcf.png)

</div>

</div>

<div class="cell markdown" id="4PVGXngCX5Sf">

*Nhận xét: - Nhóm A,B,C chủ yếu là người "ever_married". Trong khi đó,
hầu hết người trong nhóm D "not_ever_married"*

</div>

<div class="cell markdown" id="d2CuxIAUaQD5">

**- Fill null values**

</div>

<div class="cell code" execution_count="14"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="yNhlz9DMIADh" outputId="2632f022-7f7a-4ccd-da48-09d0f0b32f29">

``` python
# Đếm số lượng các giá trị null trong cột "Ever_Married" của các segment
print(train['Ever_Married'].isna().groupby(train['Segmentation']).sum())
```

<div class="output stream stdout">

    Segmentation
    A    34
    B    31
    C    23
    D    52
    Name: Ever_Married, dtype: int64

</div>

</div>

<div class="cell code" execution_count="15" id="qBATXLBSaG7J">

``` python
train.loc[train['Segmentation'] == 'D','Ever_Married'] = 'No'
train['Ever_Married'].fillna('Yes',inplace=True)
```

</div>

<div class="cell code" execution_count="16"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="ck1Ml6LDlmLH" outputId="ee51b23d-25e0-4091-e8d3-4acd57edeb56">

``` python
# Kiểm tra xem còn có giá trị null trong cột "Ever_Married" hay không
print("Number of null values in 'Ever_Married':(after)", train['Ever_Married'].isnull().sum())
```

<div class="output stream stdout">

    Number of null values in 'Ever_Married':(after) 0

</div>

</div>

<div class="cell code" execution_count="17" id="ZtvsUs7jb9ue">

``` python
# Turn label in to numerical categories
train.Ever_Married=pd.Categorical(train.Ever_Married,categories=['No','Yes'],ordered=True).codes
```

</div>

<div class="cell markdown" id="z88pAqHN-S_D">

### 1.2.3. Cột "Age"

</div>

<div class="cell code" execution_count="18"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="Z_DrPk5U-XpI" outputId="57bb55b7-c0ee-47b3-bd6e-724a2d94605d">

``` python
train['Age'].values
```

<div class="output execute_result" execution_count="18">

    array([22, 38, 67, ..., 33, 27, 37])

</div>

</div>

<div class="cell code" execution_count="19"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:570}"
id="I4A-ww7bAHeC" outputId="4d85d08e-a670-4b7c-e6e5-9f7a21bafcdd">

``` python
# Looking the distribution of column Age with respect to each segment
a = train[train.Segmentation =='A']["Age"]
b = train[train.Segmentation =='B']["Age"]
c = train[train.Segmentation =='C']["Age"]
d = train[train.Segmentation =='D']["Age"]

plt.figure(figsize=(15,5))

# Creating a boxplot
plt.subplot(1,2,1)
sns.boxplot(x='Segmentation', y='Age', data=train, order = ['A','B','C','D'])
plt.xlabel('Segmentation')
plt.ylabel('Age')
plt.title('Boxplot: Age by Segmentation')
plt.ylim(0, 100)

# Creating a kde plot
plt.subplot(1,2,2)
sns.kdeplot(a,fill = False, label = 'A')
sns.kdeplot(b,fill = False, label = 'B')
sns.kdeplot(c,fill = False, label = 'C')
sns.kdeplot(d,fill = False, label = 'D')
plt.xlabel('Age')
plt.ylabel('Density')
plt.title("Mean\n A: {}\n B: {}\n C: {}\n D: {}".format(round(a.mean(),0),round(b.mean(),0),round(c.mean(),0),round(d.mean(),0)))
plt.legend()

plt.show()
```

<div class="output display_data">

![](f19815a212381c2eea643b91c26f63c433430d89.png)

</div>

</div>

<div class="cell markdown" id="wA6OiCgLDoW7">

Nhận xét:

-   Độ tuổi trong segment D thấp nhất nhưng cũng có nhiều outliers tập
    trung từ 60 đến 90.
-   Độ tuổi trong segment A có outliers từ 80 đến 90.
-   Độ tuổi trong segment B và C khá tương đồng nhau, phân phối tuổi
    đồng nhất hơn và không có outliers.

</div>

<div class="cell markdown" id="YdOBtEaEE2_T">

### 1.2.4. Cột "Graduated"

</div>

<div class="cell code" execution_count="20"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="87_KiZw_AV2n" outputId="eadf996f-37f5-49aa-af88-0360b6a399b2">

``` python
# Các giá trị trong cột 'Graduated'
print("null values:", train.Graduated.isnull().sum())
print(train.Graduated.value_counts())
```

<div class="output stream stdout">

    null values: 78
    Yes    4968
    No     3022
    Name: Graduated, dtype: int64

</div>

</div>

<div class="cell code" execution_count="21"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:572}"
id="YiAecaxkHY91" outputId="91a62758-c561-4720-dd3d-c2bcc289b409">

``` python
# Visualization
counts = train.groupby(['Graduated', 'Segmentation']).size()
counts_df = counts.unstack(level=0)
counts_df.plot(kind='bar')

# Thêm nhãn vầ title
plt.xlabel('Segment')
plt.ylabel('Counts')
plt.title(str(counts_df))
plt.show()
```

<div class="output display_data">

![](bc4d7c672395ce65ce128591465c982f767566f6.png)

</div>

</div>

<div class="cell markdown" id="KAbmiHXAIwvz">

*Nhận xét: Segment A,B,C phần lớn là người 'Graduated'. Segment D phần
lớn là người 'Not Graduated'*

</div>

<div class="cell markdown" id="0oduKOlDJDsZ">

**- Fill null values**

</div>

<div class="cell code" execution_count="22"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="d0wrJ6ohHy_O" outputId="6ccaff18-cfb4-434a-8ebc-6686045ec15d">

``` python
# Đếm số lượng các giá trị null trong cột "Graduated" của các segment
print(train['Graduated'].isna().groupby(train['Segmentation']).sum())
```

<div class="output stream stdout">

    Segmentation
    A    24
    B    18
    C    15
    D    21
    Name: Graduated, dtype: int64

</div>

</div>

<div class="cell code" execution_count="23" id="SOGHzglGJJs0">

``` python
train.loc[train['Segmentation'] == 'D','Graduated'] = 'No'
train.Graduated.fillna('Yes',inplace=True)
```

</div>

<div class="cell code" execution_count="24"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="BK1W-0aZJmDx" outputId="e97c02dd-aa04-4e48-999b-aa5f4742aaaa">

``` python
# Kiểm tra xem còn có giá trị null trong cột "Graduated" hay không
print("Number of null values in 'Graduated'(after):", train['Graduated'].isnull().sum())
```

<div class="output stream stdout">

    Number of null values in 'Graduated'(after): 0

</div>

</div>

<div class="cell code" execution_count="25" id="E6WzwJoFcD2u">

``` python
# Turn label in to numerical categories
train.Graduated = pd.Categorical(train.Graduated,categories=['No','Yes'],ordered=True).codes
```

</div>

<div class="cell markdown" id="k3c4vafXJwH0">

### 1.2.5. Cột "Profession"

</div>

<div class="cell code" execution_count="26"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="LPLaoOcxJ1Vh" outputId="c6439324-6251-47e6-ac75-8310146fd9c5">

``` python
# Các giá trị trong cột "Profession"
print("null values:", train.Profession.isnull().sum())
print(train.Profession.value_counts())
```

<div class="output stream stdout">

    null values: 124
    Artist           2516
    Healthcare       1332
    Entertainment     949
    Engineer          699
    Doctor            688
    Lawyer            623
    Executive         599
    Marketing         292
    Homemaker         246
    Name: Profession, dtype: int64

</div>

</div>

<div class="cell code" execution_count="27"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="5_s81EYzKkCS" outputId="d31d2145-aea1-40da-b2a3-ad608521dd69">

``` python
value_counts = train.groupby('Segmentation')['Profession'].value_counts()
value_counts.groupby(level=0, group_keys=False).nlargest(len(value_counts))
```

<div class="output execute_result" execution_count="27">

    Segmentation  Profession   
    A             Artist            558
                  Entertainment     365
                  Engineer          259
                  Doctor            199
                  Lawyer            197
                  Executive         125
                  Healthcare        106
                  Homemaker          73
                  Marketing          57
    B             Artist            756
                  Entertainment     221
                  Engineer          189
                  Executive         183
                  Lawyer            158
                  Doctor            143
                  Healthcare        101
                  Homemaker          55
                  Marketing          30
    C             Artist           1065
                  Executive         175
                  Entertainment     148
                  Healthcare        146
                  Doctor            140
                  Lawyer            140
                  Engineer           75
                  Marketing          35
                  Homemaker          28
    D             Healthcare        979
                  Entertainment     215
                  Doctor            206
                  Engineer          176
                  Marketing         170
                  Artist            137
                  Lawyer            128
                  Executive         116
                  Homemaker          90
    Name: Profession, dtype: int64

</div>

</div>

<div class="cell code" execution_count="28"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:907}"
id="K0IzlNZiSBzF" outputId="a6b70677-99d8-4098-9c98-0c29dbf7465e">

``` python
# Tạo từ điển ánh xạ giữa segment và màu tương ứng
segment_colors = {
    'A': 'blue',
    'B': 'green',
    'C': 'orange',
    'D': 'red'
}

# Tạo subplot với 3 hàng và 3 cột
fig, axs = plt.subplots(3, 3, figsize=(9, 9))

# Lấy danh sách các giá trị trong cột Profession và sắp xếp theo counts
professions = train['Profession'].value_counts().index

# Lấy danh sách các giá trị trong cột Segmentation
segments = train['Segmentation'].unique()

# Duyệt qua từng giá trị trong cột Profession và vẽ plot tương ứng
for i, profession in enumerate(professions):
    row = i // 3
    col = i % 3
    ax = axs[row, col]

    # Lọc dữ liệu theo giá trị của Profession
    data = train[train['Profession'] == profession]

    # Vẽ biểu đồ cột trong plot hiện tại
    for j, segment in enumerate(segments):
        segment_data = data[data['Segmentation'] == segment]
        ax.bar(segment, segment_data.shape[0], color=segment_colors[segment])

    # Đặt tiêu đề cho plot
    ax.set_title(f'{profession}')

# Cân chỉnh và hiển thị subplot
plt.tight_layout()
plt.show()
```

<div class="output display_data">

![](9eba90887b3df0a18b3a351da954fa7da2c9ae9f.png)

</div>

</div>

<div class="cell markdown" id="bjTLelfrSfKG">

*Nhận xét: Nhóm A, B, C nhiều nhất là Artist, D nhiều nhất là
Healthcare*

</div>

<div class="cell markdown" id="fsVLIJxeS1X1">

**- Fill null values**

</div>

<div class="cell code" execution_count="29"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="6qSiyMARSxcL" outputId="b69f86c4-d4e0-4921-d43a-16325e5199b8">

``` python
# Đếm số lượng các giá trị null trong cột "Profession" của các segment
print(train['Profession'].isna().groupby(train['Segmentation']).sum())
```

<div class="output stream stdout">

    Segmentation
    A    33
    B    22
    C    18
    D    51
    Name: Profession, dtype: int64

</div>

</div>

<div class="cell code" execution_count="30" id="uOrvA8cbTbPh">

``` python
train.loc[train['Segmentation'] == 'D','Profession'] = 'Healthcare'
train.Profession.fillna('Artist',inplace=True)
```

</div>

<div class="cell code" execution_count="31"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="pomHE9NXTf8f" outputId="7893ed7a-b132-4f47-d81a-535188566d0d">

``` python
# Kiểm tra xem còn có giá trị null trong cột "Profession" hay không
print("Number of null values in 'Profession'(after):", train['Profession'].isnull().sum())
# Các giá trị trong cột "Profession"
print(train.Profession.value_counts())
```

<div class="output stream stdout">

    Number of null values in 'Profession'(after): 0
    Healthcare       2621
    Artist           2452
    Entertainment     734
    Engineer          523
    Lawyer            495
    Executive         483
    Doctor            482
    Homemaker         156
    Marketing         122
    Name: Profession, dtype: int64

</div>

</div>

<div class="cell code" execution_count="32" id="M5Ra67ijcK1a">

``` python
# Turn label in to numerical categories
train.Profession=pd.Categorical(train.Profession,categories=['Homemaker', 'Artist', 'Healthcare', 'Entertainment', 'Doctor', 'Lawyer', 'Executive', 'Marketing', 'Engineer'],ordered=True).codes
```

</div>

<div class="cell markdown" id="RAReVU1uUTrv">

### 1.2.6 Cột "Speding_Score"

</div>

<div class="cell code" execution_count="33"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="sYwHTu8LUbhP" outputId="bf9401d3-1fb1-4d90-f0e4-a5f59811e320">

``` python
# Đếm số lượng các giá trị null trong cột "Spending_Score" của các segment
count_ss = train.groupby(["Segmentation"])["Spending_Score"].value_counts().unstack()
print(count_ss)
```

<div class="output stream stdout">

    Spending_Score  Average  High   Low
    Segmentation                       
    A                   343   271  1358
    B                   590   384   884
    C                   903   405   662
    D                   138   156  1974

</div>

</div>

<div class="cell code" execution_count="34"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:573}"
id="eI6OUKEJaG1W" outputId="fe583086-574d-4398-ed09-9175d0e8a067">

``` python
# Visualize
count_ss.plot(kind = 'bar')
plt.title(str(count_ss))
plt.show()
```

<div class="output display_data">

![](f046bfb2205b8672892a6faf1e2ebdb2705fe4f5.png)

</div>

</div>

<div class="cell code" execution_count="35" id="cOWJRqnRalt_">

``` python
# Turn label in to numerical categories
train.Spending_Score=pd.Categorical(train.Spending_Score,categories=['Low','Average','High'],ordered=True).codes
```

</div>

<div class="cell markdown" id="k8OE7SVmdRrt">

### 1.2.7. Cột "Work_Experience"

</div>

<div class="cell code" execution_count="36"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="giiIUvnOdV3A" outputId="a088719e-2ea0-489b-ef18-a2866a944943">

``` python
# Các giá trị trong cột "Work_Experience"
print("null values:", train.Work_Experience.isnull().sum())
print(train.Work_Experience.value_counts())
```

<div class="output stream stdout">

    null values: 829
    1.0     2354
    0.0     2318
    9.0      474
    8.0      463
    2.0      286
    3.0      255
    4.0      253
    6.0      204
    7.0      196
    5.0      194
    10.0      53
    11.0      50
    12.0      48
    13.0      46
    14.0      45
    Name: Work_Experience, dtype: int64

</div>

</div>

<div class="cell code" execution_count="37"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:570}"
id="cC4go1gCMl5i" outputId="237ea4a7-0495-46d4-de2f-effe0b53f85c">

``` python
# Looking the distribution of column Work_Experience w.r.t to each segment
a = train[train.Segmentation =='A']["Work_Experience"]
b = train[train.Segmentation =='B']["Work_Experience"]
c = train[train.Segmentation =='C']["Work_Experience"]
d = train[train.Segmentation =='D']["Work_Experience"]

plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
sns.boxplot(data = train, x = "Segmentation", y="Work_Experience", order = ['A','B','C','D'])
plt.title('Boxplot')

plt.subplot(1,2,2)
sns.kdeplot(a,fill = False, label = 'A')
sns.kdeplot(b,fill = False, label = 'B')
sns.kdeplot(c,fill = False, label = 'C')
sns.kdeplot(d,fill = False, label = 'D')
plt.xlabel('Work Experience')
plt.ylabel('Density')
plt.title("Mean\n A: {}\n B: {}\n C: {}\n D: {}".format(round(a.mean(),0),round(b.mean(),0),round(c.mean(),0),round(d.mean(),0)))

plt.show()
```

<div class="output display_data">

![](9d35897a674908b2bec9c5b11a43338dbdabb776.png)

</div>

</div>

<div class="cell code" execution_count="38"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="IGy51fiFNHdC" outputId="ed0d80d9-3593-44b8-f76b-ceed6c8be003">

``` python
# Đếm số lượng các giá trị null trong cột "Work_Experience" của các segment
print(train['Work_Experience'].isna().groupby(train['Segmentation']).sum())
```

<div class="output stream stdout">

    Segmentation
    A    194
    B    192
    C    155
    D    288
    Name: Work_Experience, dtype: int64

</div>

</div>

<div class="cell markdown" id="SWND-y1iNnNQ">

*Cột này sẽ được bỏ đi vì dữ liệu sẽ không giúp ích nhiều*

</div>

<div class="cell code" execution_count="39" id="UlTE5JyANuOx">

``` python
### 1.3.8. Cột "Family_Size"
```

</div>

<div class="cell code" execution_count="40"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="6ndPQM3EN4Zd" outputId="bcc78b89-5d38-43b0-97f1-8739172dd43f">

``` python
# Các giá trị trong cột "Family_Size"
print("null values:", train.Family_Size.isnull().sum())
print(train.Family_Size.value_counts())
```

<div class="output stream stdout">

    null values: 335
    2.0    2390
    3.0    1497
    1.0    1453
    4.0    1379
    5.0     612
    6.0     212
    7.0      96
    8.0      50
    9.0      44
    Name: Family_Size, dtype: int64

</div>

</div>

<div class="cell code" execution_count="41"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:570}"
id="0O1lWufqOWht" outputId="fd404f90-d183-4c33-c634-2be8bf2caf5f">

``` python
# Looking the distribution of column Family Size w.r.t to each segment
a = train[train.Segmentation =='A']["Family_Size"]
b = train[train.Segmentation =='B']["Family_Size"]
c = train[train.Segmentation =='C']["Family_Size"]
d = train[train.Segmentation =='D']["Family_Size"]

plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
sns.boxplot(data = train, x = "Segmentation", y="Family_Size", order = ['A','B','C','D'])
plt.title('Boxplot')

plt.subplot(1,2,2)
sns.kdeplot(a, fill = False, label = 'A')
sns.kdeplot(b, fill = False, label = 'B')
sns.kdeplot(c, fill = False, label = 'C')
sns.kdeplot(d, fill = False, label = 'D')
plt.xlabel('Family Size')
plt.ylabel('Density')
plt.title("Mean\n A: {}\n B: {}\n C: {}\n D: {}".format(round(a.mean(),0),round(b.mean(),0),round(c.mean(),0),round(d.mean(),0)))
plt.legend()

plt.show()
```

<div class="output display_data">

![](bba0286ea9474277db87def331d2c4e86f4ba2ba.png)

</div>

</div>

<div class="cell markdown" id="vAwSGJpyPPjd">

*Nhận xét: 'Family_Size' của B,C,D khá tương đồng nhau, trong khi của A
thấp hơn.*

</div>

<div class="cell markdown" id="vdj-BSt4QBQZ">

**- Fill null values**

</div>

<div class="cell code" execution_count="42"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="gifuxnSXPeyZ" outputId="dc4c28f8-f4f3-491b-b2df-fccddb187315">

``` python
# Đếm số lượng các giá trị null trong cột "Family_Size" của các segment
print(train['Family_Size'].isna().groupby(train['Segmentation']).sum())
```

<div class="output stream stdout">

    Segmentation
    A     95
    B     43
    C     44
    D    153
    Name: Family_Size, dtype: int64

</div>

</div>

<div class="cell code" execution_count="43" id="1HBVbOJkQSv2">

``` python
train.Family_Size.fillna(2,inplace=True)
```

</div>

<div class="cell code" execution_count="44"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="WWoml93sQ0AS" outputId="b3c264be-bded-4b11-dbec-e4bddfb53563">

``` python
# Kiểm tra xem còn có giá trị null trong cột "Family_Size" hay không
print("null values:", train.Family_Size.isnull().sum())
# Các giá trị trong cột "Family_Size"
print(train.Family_Size.value_counts())
```

<div class="output stream stdout">

    null values: 0
    2.0    2725
    3.0    1497
    1.0    1453
    4.0    1379
    5.0     612
    6.0     212
    7.0      96
    8.0      50
    9.0      44
    Name: Family_Size, dtype: int64

</div>

</div>

<div class="cell markdown" id="afc-ktNiR5Ak">

### 1.2.8. Cột "Var_1"

</div>

<div class="cell code" execution_count="45"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="wgeR0pDbR9ih" outputId="15edc8cc-da9f-4985-f660-6dfb7e9228e0">

``` python
# Các giá trị trong cột "Var_1"
print("null values:", train.Var_1.isnull().sum())
print(train.Var_1.value_counts())
```

<div class="output stream stdout">

    null values: 76
    Cat_6    5238
    Cat_4    1089
    Cat_3     822
    Cat_2     422
    Cat_7     203
    Cat_1     133
    Cat_5      85
    Name: Var_1, dtype: int64

</div>

</div>

<div class="cell code" execution_count="46"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:514}"
id="nuTBiILSSJIg" outputId="bc1f1571-8e11-4f7b-84f0-481df39e83d4">

``` python
# Counting Var_1 in each segment
ax1 = train.groupby(["Segmentation"])["Var_1"].value_counts().unstack().round(3)

# Percentage of category of Var_1 in each segment
ax2 = train.pivot_table(columns='Var_1',index='Segmentation',values='ID',aggfunc='count')
ax2 = ax2.div(ax2.sum(axis=1), axis = 0).round(2)

#count plot
fig, ax = plt.subplots(1,2)
ax1.plot(kind="bar",ax = ax[0],figsize = (15,4))
ax[0].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)
ax[0].set_title(str(ax1))

#stacked bars
ax2.plot(kind="bar",stacked = True,ax = ax[1],figsize = (15,4))
ax[1].set_xticklabels(labels = ['A','B','C','D'],rotation = 0)
ax[1].set_title(str(ax2))
plt.show()
```

<div class="output display_data">

![](186c78c6e197b3bb538f11827f0add52d955b7ad.png)

</div>

</div>

<div class="cell markdown" id="HuKFizwUSeQF">

*Nhận xét: Cat_6 chiếm phần lớn trong tất cả các segment*

</div>

<div class="cell markdown" id="XuuXiSntSrFg">

**- Chuyển tất cả các giá trị null thành Cat_6**

</div>

<div class="cell code" execution_count="47" id="jRKIFfCzSo6T">

``` python
train.Var_1.fillna('Cat_6',inplace=True)
```

</div>

<div class="cell code" execution_count="48" id="fFcTeVJOS0Jg">

``` python
train.Var_1 = pd.Categorical(train.Var_1,categories=['Cat_1', 'Cat_2', 'Cat_3', 'Cat_4', 'Cat_5', 'Cat_6', 'Cat_7'],ordered=True).codes
```

</div>

<div class="cell markdown" id="6OTukIygS7N0">

## 1.3 Feature Engineering

</div>

<div class="cell code" execution_count="49" id="mJEGwU1pTf-t">

``` python
# Chuyển cột 'Segmentation' sang numerical
train.Segmentation = pd.Categorical (train.Segmentation, categories = ['A', 'B', 'C', 'D'], ordered = True).codes
```

</div>

<div class="cell code" execution_count="50" id="Gf5EggSLTh2L">

``` python
# Drop cột 'ID'
train.drop('ID', axis = 1, inplace = True)
```

</div>

<div class="cell code" execution_count="51"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:911}"
id="aG1wrBGlS-9f" outputId="73e25a85-62f1-42fc-98c9-a85a5c3e748d">

``` python
# bảng correlation giữa các cột vs nhau
cor = train.corr(method='pearson')

# select features that have high absolute correlation with output.
fig, ax = plt.subplots(figsize=(11,11))         # Sample figsize in inches
sns.heatmap(
    cor, #dataset
    vmin=-1, vmax=1, #values to anchor the heatmap
    center=0, #value để center, look at the color bar
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True, #each cell is square-shaped
    ax=ax,
    annot=True #để mỗi cell có text
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
```

<div class="output display_data">

![](2a5fc936280792790db2b886a7ca30b586a1a875.png)

</div>

</div>

<div class="cell code" execution_count="52" id="F_sk6GLlUFuM">

``` python
# Corr của work experience, var 1, gender < 0,1 -> bỏ
train = train.drop ('Work_Experience', axis =1)
train = train.drop ('Var_1', axis = 1)
train = train.drop ('Gender', axis = 1)
```

</div>

<div class="cell code" execution_count="53"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:424}"
id="SSvhVopjUPxk" outputId="4518573a-e476-4ec2-dba0-25e8068e9e20">

``` python
# Xem dữ liệu trong 'train' sau khi chuyển đổi sang numerical
train
```

<div class="output execute_result" execution_count="53">

          Ever_Married  Age  Graduated  Profession  Spending_Score  Family_Size  \
    0                0   22          0           2               0          4.0   
    1                1   38          1           8               1          3.0   
    2                1   67          1           8               0          1.0   
    3                1   67          1           5               2          2.0   
    4                1   40          1           3               2          6.0   
    ...            ...  ...        ...         ...             ...          ...   
    8063             0   22          0           2               0          7.0   
    8064             0   35          0           2               0          4.0   
    8065             0   33          0           2               0          1.0   
    8066             0   27          1           2               0          4.0   
    8067             1   37          1           6               1          3.0   

          Segmentation  
    0                3  
    1                0  
    2                1  
    3                1  
    4                0  
    ...            ...  
    8063             3  
    8064             3  
    8065             3  
    8066             1  
    8067             1  

    [8068 rows x 7 columns]

</div>

</div>

<div class="cell markdown" id="o6MjL-TKVhm3">

**=> Từ phần 1, ta có một bộ dữ liệu numerical dựa trên 'train_dataset',
ta sẽ gán dữ liệu này là 'data'**

</div>

<div class="cell code" execution_count="54" id="urn-cf4bVs8H">

``` python
data = train.copy()
```

</div>

<div class="cell code" execution_count="55"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="H1342ug0Wsll" outputId="68590ee7-8b38-446d-b109-792cb2adee1b">

``` python
data.info()
```

<div class="output stream stdout">

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8068 entries, 0 to 8067
    Data columns (total 7 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   Ever_Married    8068 non-null   int8   
     1   Age             8068 non-null   int64  
     2   Graduated       8068 non-null   int8   
     3   Profession      8068 non-null   int8   
     4   Spending_Score  8068 non-null   int8   
     5   Family_Size     8068 non-null   float64
     6   Segmentation    8068 non-null   int8   
    dtypes: float64(1), int64(1), int8(5)
    memory usage: 165.6 KB

</div>

</div>

<div class="cell markdown" id="fVtRGJxDW21Q">

# **2. Phân loại dữ liệu bằng các mô hình học máy**

</div>

<div class="cell markdown" id="q6cijQHcnY52">

## 2.1. Thử các mô hình học máy

</div>

<div class="cell markdown" id="MlggK1sJlvQo">

Dùng các model: *Softmax Regression, KNeighborsClassifiers,
LGMCLassifier, DecisionTreeClassifier, RandomForestClassifier,
SupportVectorMachineClassifier, NaiveBayesCLassifier* để phân loại dữ
liệu train. Sau đó chọn mô hình tốt nhất để tối ưu.

</div>

<div class="cell code" execution_count="56" id="1X8-BDZMYiU_">

``` python
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.metrics import accuracy_score

from lightgbm  import LGBMClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble    import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
```

</div>

<div class="cell markdown" id="Udcn6u5sXKjF">

Chia dữ liệu 'data' thành X_train, y_train, X_val, y_val

</div>

<div class="cell code" execution_count="57"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="cy5AVQCwW5P6" outputId="72d8fe63-e682-4ce9-e7b3-9a408ebeb9bb">

``` python
X = data.values[:,:6]   # X.shape = (8068,8)
y = data.values[:,6]    # Y.shape = (8068,)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)

print('X_train.shape:', X_train.shape)
print('y_train.shape:', y_train.shape)
print('X_val.shape:', X_val.shape)
print('y_val.shape:', y_val.shape)
```

<div class="output stream stdout">

    X_train.shape: (6454, 6)
    y_train.shape: (6454,)
    X_val.shape: (1614, 6)
    y_val.shape: (1614,)

</div>

</div>

<div class="cell code" execution_count="58" id="Eq8iKqS_fxh6">

``` python
# k-fold
num_folds = 10
seed = 42
scoring = 'accuracy'
```

</div>

<div class="cell code" execution_count="59" id="n1RMSnY6f7JR">

``` python
models = []
models.append(('LR', LogisticRegression(multi_class = 'multinomial', max_iter = 10000)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('LGB', LGBMClassifier()))
models.append(('Decision tree', DecisionTreeClassifier(max_depth = 10)))
models.append(('Random Forest', RandomForestClassifier()))
models.append(('SVC', SVC(decision_function_shape = 'ovo')))
models.append(('Naive Bayes', GaussianNB()))
```

</div>

<div class="cell code" execution_count="60"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="wLk9R7eNgVnh" outputId="18dd176b-9f1f-40f2-b5ec-a56c08c90325">

``` python
# evaluate each model in turn
results = []
names = []
for name, model in models:
 kfold = model_selection.KFold(n_splits=10, shuffle = True, random_state=seed)
 cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
 results.append(cv_results)
 names.append(name)
 msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
 print(msg)
```

<div class="output stream stdout">

    LR: 0.608205 (0.018858)
    KNN: 0.598660 (0.020080)
    LGB: 0.650099 (0.012623)
    Decision tree: 0.640554 (0.016703)
    Random Forest: 0.621221 (0.016473)
    SVC: 0.539043 (0.021807)
    Naive Bayes: 0.622831 (0.018413)

</div>

</div>

<div class="cell code" execution_count="61"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:1000}"
id="DN1sO0Mpgn-8" outputId="a5a67533-4284-4b5d-c9d4-808979e26dcb">

``` python
fig = plt.figure(figsize=(15, 15))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
```

<div class="output display_data">

![](c6368fd38000c4279747d0e5b7e2ac03793ec874.png)

</div>

</div>

<div class="cell code" execution_count="62"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="UCBy2-QnhUiW" outputId="13dd0245-4c55-4844-a20b-897972064400">

``` python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import *

pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LogisticRegression(multi_class = 'multinomial', max_iter=1000))])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('Decision Tree',DecisionTreeClassifier())])))
pipelines.append(('ScaledLGB', Pipeline([('Scaler', StandardScaler()),('LGB', LGBMClassifier())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('Random Forest', RandomForestClassifier())])))
pipelines.append(('ScaledSVC', Pipeline([('Scaler', StandardScaler()),('SVC', SVC(decision_function_shape='ovo'))])))
pipelines.append(('ScaledNaive', Pipeline([('Scaler', StandardScaler()),('Naive Bayes', GaussianNB())])))


results = []
names = []
for name, model in pipelines:
  kfold = KFold(n_splits=num_folds, shuffle = True, random_state=seed)
  cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)
```

<div class="output stream stdout">

    ScaledLR: 0.608765 (0.012875)
    ScaledKNN: 0.614347 (0.018594)
    ScaledCART: 0.606137 (0.011501)
    ScaledLGB: 0.644250 (0.014573)
    ScaledKNN: 0.620856 (0.012110)
    ScaledSVC: 0.634487 (0.011246)
    ScaledNaive: 0.619613 (0.015024)

</div>

</div>

<div class="cell code" execution_count="63"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:1000}"
id="YgOO7OMojQ9c" outputId="9a896854-10aa-4cf2-a029-0d040be298f9">

``` python
# Compare Algorithms đã chuẩn hóa
fig = plt.figure(figsize=(12, 12))
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
```

<div class="output display_data">

![](134d7c0e4c2715eecacbfe8cd806a5e614aed609.png)

</div>

</div>

<div class="cell markdown" id="F-u8GmZWfyTx">

#### => Ở cả 'Not_Scaled_Data' và 'Scaled_Data', LGBMClassifier đều cho kết quả tốt nhất. Do đó, ta sẽ chọn tối ưu hóa mô hình này

</div>

<div class="cell markdown" id="UY-syNe1fyNd">

## 2.2. Tối ưu hóa LGBMClassifier

</div>

<div class="cell code" execution_count="64"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="lnnIxjExpgl_" outputId="1ed78302-d249-4039-c749-a1be928c365b">

``` python
from sklearn.model_selection import RandomizedSearchCV

model = LGBMClassifier()
parameters = {'learning_rate': [0.01], 'n_estimators': [8, 24],
    'num_leaves':[20,40,60,80,100], 'min_child_samples':[5,10,15],'max_depth':[-1,5,10,20],
             'learning_rate':[0.05,0.1,0.2],'reg_alpha':[0,0.01,0.03], 'colsample_bytree': [0.65, 0.75, 0.8],}
clf = RandomizedSearchCV(model, parameters, scoring = 'accuracy', n_iter=100)
clf.fit(X = X_train, y = y_train)
print(clf.best_params_)
predicted = clf.predict(X_val)
print('Classification of the result is:')
print(accuracy_score(y_val, predicted))
```

<div class="output stream stdout">

    {'reg_alpha': 0.03, 'num_leaves': 40, 'n_estimators': 8, 'min_child_samples': 5, 'max_depth': -1, 'learning_rate': 0.05, 'colsample_bytree': 0.75}
    Classification of the result is:
    0.6617100371747212

</div>

</div>

<div class="cell markdown" id="5lmMtsFtsEut">

=> Tham số tốt nhất là: *{'reg_alpha': 0.03, 'num_leaves': 20,
'n_estimators': 24, 'min_child_samples': 10, 'max_depth': 10,
'learning_rate': 0.1, 'colsample_bytree': 0.65}*

</div>

<div class="cell code" execution_count="65"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:92}"
id="D7IHNROaqd8R" outputId="d114d83b-45dc-4896-d8e2-0376f1af9795">

``` python
best_lgbm = LGBMClassifier(reg_alpha=0.03, num_leaves=20, n_estimators=24, min_child_samples=10, max_depth=10, learning_rate=0.1, colsample_bytree=0.65)
best_lgbm.fit(X_train, y_train)
```

<div class="output execute_result" execution_count="65">

    LGBMClassifier(colsample_bytree=0.65, max_depth=10, min_child_samples=10,
                   n_estimators=24, num_leaves=20, reg_alpha=0.03)

</div>

</div>

<div class="cell code" execution_count="66"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="OrW6ZZxusWrT" outputId="9d6a6231-5cf3-4a48-b5bf-5cf83fe0aeb7">

``` python
from sklearn.metrics import *

predictions = best_lgbm.predict(X_val)
print (classification_report(y_val, predictions))
```

<div class="output stream stdout">

                  precision    recall  f1-score   support

             0.0       0.56      0.68      0.62       391
             1.0       0.47      0.26      0.33       369
             2.0       0.54      0.62      0.58       380
             3.0       0.95      1.00      0.98       474

        accuracy                           0.66      1614
       macro avg       0.63      0.64      0.63      1614
    weighted avg       0.65      0.66      0.65      1614

</div>

</div>

<div class="cell code" execution_count="67"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="yk0HiQnMss0B" outputId="5e90dd7b-b277-4aa7-d80a-708408152fbd">

``` python
cm = confusion_matrix(y_val, predictions)
print(cm)
```

<div class="output stream stdout">

    [[265  54  68   4]
     [129  96 136   8]
     [ 76  55 237  12]
     [  0   0   0 474]]

</div>

</div>

<div class="cell code" execution_count="68"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:738}"
id="SfkDMQ7IsxFO" outputId="3d04c989-2a22-4bb3-f821-96b92b5fb644">

``` python
# Visualize bằng seaborn
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt="n", linewidths=.5, square = True, cmap = 'bone_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(accuracy_score(y_val, predicted))
plt.title(all_sample_title, size = 15);
```

<div class="output display_data">

![](dbf4f6db46ade39e3691077839030b38d691a967.png)

</div>

</div>

<div class="cell code" execution_count="69"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:253}"
id="UKXBb0jbuO9a" outputId="06afc6d2-792a-4668-9c9a-acbcf0f887e5">

``` python
import lightgbm as lgb
from IPython.display import Image
import pydotplus

# Tạo đồ thị cây quyết định
graph = lgb.create_tree_digraph(best_lgbm, tree_index=3, name='Tree3')

# Chuyển đổi đồ thị thành hình ảnh
image = pydotplus.graph_from_dot_data(graph.source).create_png()

# Hiển thị hình ảnh
Image(image)
```

<div class="output execute_result" execution_count="69">

![](95602734c71a494487102bc15639ffc6ca86d97e.png)

</div>

</div>

<div class="cell markdown" id="2yYwb2ngcbEq">

# **3. Phân loại dữ liệu bằng mô hình Deep Learning**

</div>

<div class="cell code" execution_count="70"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="-DpXGDMe0ugs" outputId="4bab7ad2-a749-497b-c717-2cd2ea52e795">

``` python
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.layers import Dropout

# Mã hóa nhãn y
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)

# Chuyển đổi nhãn y sang dạng one-hot
num_classes = len(label_encoder.classes_)
y_train_onehot = np_utils.to_categorical(y_train_encoded, num_classes)
y_val_onehot = np_utils.to_categorical(y_val_encoded, num_classes)

# Xây dựng kiến trúc mô hình
model_DL = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(6,)),
    Dropout(0.5),  # Thêm dropout với tỷ lệ loại bỏ 50%
    keras.layers.Dense(64, activation='relu'),
    Dropout(0.5),  # Thêm dropout với tỷ lệ loại bỏ 50%
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compile mô hình
model_DL.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model_DL.fit(X_train, y_train_onehot, epochs=20, batch_size=32, validation_data=(X_val, y_val_onehot))

# Đánh giá mô hình
test_loss, test_acc = model_DL.evaluate(X_val, y_val_onehot)
print('Test accuracy:', test_acc)
```

<div class="output stream stdout">

    Epoch 1/20
    202/202 [==============================] - 2s 5ms/step - loss: 3.1177 - accuracy: 0.2611 - val_loss: 1.3822 - val_accuracy: 0.2937
    Epoch 2/20
    202/202 [==============================] - 1s 3ms/step - loss: 1.4891 - accuracy: 0.2842 - val_loss: 1.3814 - val_accuracy: 0.2937
    Epoch 3/20
    202/202 [==============================] - 1s 4ms/step - loss: 1.4102 - accuracy: 0.3026 - val_loss: 1.3788 - val_accuracy: 0.2937
    Epoch 4/20
    202/202 [==============================] - 1s 4ms/step - loss: 1.3752 - accuracy: 0.3215 - val_loss: 1.3488 - val_accuracy: 0.3569
    Epoch 5/20
    202/202 [==============================] - 1s 3ms/step - loss: 1.3335 - accuracy: 0.3458 - val_loss: 1.2752 - val_accuracy: 0.4294
    Epoch 6/20
    202/202 [==============================] - 1s 3ms/step - loss: 1.2694 - accuracy: 0.4022 - val_loss: 1.1300 - val_accuracy: 0.5533
    Epoch 7/20
    202/202 [==============================] - 1s 3ms/step - loss: 1.1786 - accuracy: 0.4371 - val_loss: 1.0067 - val_accuracy: 0.5161
    Epoch 8/20
    202/202 [==============================] - 1s 3ms/step - loss: 1.0859 - accuracy: 0.4782 - val_loss: 0.9380 - val_accuracy: 0.5266
    Epoch 9/20
    202/202 [==============================] - 1s 4ms/step - loss: 1.0194 - accuracy: 0.5023 - val_loss: 0.9006 - val_accuracy: 0.5551
    Epoch 10/20
    202/202 [==============================] - 1s 5ms/step - loss: 0.9764 - accuracy: 0.5191 - val_loss: 0.8877 - val_accuracy: 0.5582
    Epoch 11/20
    202/202 [==============================] - 1s 6ms/step - loss: 0.9519 - accuracy: 0.5327 - val_loss: 0.8737 - val_accuracy: 0.6059
    Epoch 12/20
    202/202 [==============================] - 1s 4ms/step - loss: 0.9472 - accuracy: 0.5389 - val_loss: 0.8691 - val_accuracy: 0.6047
    Epoch 13/20
    202/202 [==============================] - 1s 3ms/step - loss: 0.9234 - accuracy: 0.5535 - val_loss: 0.8587 - val_accuracy: 0.6016
    Epoch 14/20
    202/202 [==============================] - 1s 3ms/step - loss: 0.9180 - accuracy: 0.5530 - val_loss: 0.8523 - val_accuracy: 0.6159
    Epoch 15/20
    202/202 [==============================] - 1s 3ms/step - loss: 0.9101 - accuracy: 0.5615 - val_loss: 0.8482 - val_accuracy: 0.6072
    Epoch 16/20
    202/202 [==============================] - 1s 4ms/step - loss: 0.9046 - accuracy: 0.5665 - val_loss: 0.8490 - val_accuracy: 0.6059
    Epoch 17/20
    202/202 [==============================] - 1s 3ms/step - loss: 0.8980 - accuracy: 0.5669 - val_loss: 0.8378 - val_accuracy: 0.6183
    Epoch 18/20
    202/202 [==============================] - 1s 3ms/step - loss: 0.8909 - accuracy: 0.5728 - val_loss: 0.8394 - val_accuracy: 0.6134
    Epoch 19/20
    202/202 [==============================] - 1s 3ms/step - loss: 0.8771 - accuracy: 0.5827 - val_loss: 0.8325 - val_accuracy: 0.6010
    Epoch 20/20
    202/202 [==============================] - 1s 3ms/step - loss: 0.8771 - accuracy: 0.5798 - val_loss: 0.8213 - val_accuracy: 0.6190
    51/51 [==============================] - 0s 2ms/step - loss: 0.8213 - accuracy: 0.6190
    Test accuracy: 0.6189591288566589

</div>

</div>

<div class="cell markdown" id="JbAR3Y0eereu">

# **4. Sử dụng mô hình để dự đoán trên tập test**

</div>

<div class="cell markdown" id="qgAJhDPlnr23">

*Import tập dữ liệu test*

</div>

<div class="cell code" execution_count="71" id="Gikk9Vl9lGcP">

``` python
Test = pd.read_csv('Test.csv')
```

</div>

<div class="cell code" execution_count="72"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="Tla39Nua2BNR" outputId="663b7df4-2c6b-4f3f-d5dc-6e93ac8f4a2c">

``` python
Test.isnull().sum()
```

<div class="output execute_result" execution_count="72">

    ID                   0
    Gender               0
    Ever_Married        50
    Age                  0
    Graduated           24
    Profession          38
    Work_Experience    269
    Spending_Score       0
    Family_Size        113
    Var_1               32
    Segmentation         0
    dtype: int64

</div>

</div>

<div class="cell code" execution_count="73"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:424}"
id="B82BnfAwe2Hx" outputId="57f93318-0387-4515-a48c-59b54e21c025">

``` python
# Biến đổi dữ liệu tập test như tập train
Test.Gender = pd.Categorical(Test.Gender,categories=['Male','Female'],ordered=True).codes
Test.loc[Test['Segmentation'] == 'D','Ever_Married'] = 'No'
Test['Ever_Married'].fillna('Yes',inplace=True)
Test.Ever_Married=pd.Categorical(Test.Ever_Married,categories=['No','Yes'],ordered=True).codes
Test.loc[Test['Segmentation'] == 'D','Graduated'] = 'No'
Test.Graduated.fillna('Yes',inplace=True)
Test.Graduated = pd.Categorical(Test.Graduated,categories=['No','Yes'],ordered=True).codes
Test.loc[Test['Segmentation'] == 'D','Profession'] = 'Healthcare'
Test.Profession.fillna('Artist',inplace=True)
Test.Profession=pd.Categorical(Test.Profession,categories=['Homemaker', 'Artist', 'Healthcare', 'Entertainment', 'Doctor', 'Lawyer', 'Executive', 'Marketing', 'Engineer'],ordered=True).codes
Test.Spending_Score=pd.Categorical(Test.Spending_Score,categories=['Low','Average','High'],ordered=True).codes
Test.Family_Size.fillna(2,inplace=True)
Test.Var_1.fillna('Cat_6',inplace=True)
Test.Var_1 = pd.Categorical(Test.Var_1,categories=['Cat_1', 'Cat_2', 'Cat_3', 'Cat_4', 'Cat_5', 'Cat_6', 'Cat_7'],ordered=True).codes
Test.Segmentation = pd.Categorical (Test.Segmentation, categories = ['A', 'B', 'C', 'D'], ordered = True).codes
Test = Test.drop ('ID', axis =1)
Test = Test.drop ('Work_Experience', axis =1)
Test = Test.drop ('Var_1', axis = 1)
Test = Test.drop ('Gender', axis = 1)
Test
```

<div class="output execute_result" execution_count="73">

          Ever_Married  Age  Graduated  Profession  Spending_Score  Family_Size  \
    0                1   36          1           8               0          1.0   
    1                1   37          1           2               1          4.0   
    2                1   69          0           1               0          1.0   
    3                1   59          0           6               2          2.0   
    4                0   19          0           7               0          4.0   
    ...            ...  ...        ...         ...             ...          ...   
    2622             0   29          0           2               0          4.0   
    2623             0   35          1           4               0          1.0   
    2624             0   53          1           3               0          2.0   
    2625             1   47          1           6               2          5.0   
    2626             0   43          1           2               0          3.0   

          Segmentation  
    0                1  
    1                0  
    2                0  
    3                1  
    4                0  
    ...            ...  
    2622             1  
    2623             0  
    2624             2  
    2625             2  
    2626             0  

    [2627 rows x 7 columns]

</div>

</div>

<div class="cell code" execution_count="74"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="t-udfyggn_nB" outputId="aaa96004-62c9-49d5-c925-475c9eb66e24">

``` python
X_test = Test.values[:,:6]
y_test = Test.values[:,6]
print(X_test.shape)
print(y_test.shape)
```

<div class="output stream stdout">

    (2627, 6)
    (2627,)

</div>

</div>

<div class="cell markdown" id="Xu5QpBCjoDz8">

## 4.1. Sử dụng model LGBMClassifier đã tối ưu

</div>

<div class="cell code" execution_count="75" id="Kf1eMnzzlzQX">

``` python
y_pred = best_lgbm.predict(X_test)
```

</div>

<div class="cell code" execution_count="78"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="e2bz8FGAl1ik" outputId="1c201089-dea4-4ceb-e5b5-78e49db08e17">

``` python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

<div class="output stream stdout">

    Accuracy: 0.5523

</div>

</div>

<div class="cell markdown" id="AyaNqMSboJ4s">

## 4.2. Sử dụng DeepLearning

</div>

<div class="cell code" execution_count="80"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="cUwx_uJW1ReC" outputId="c24b4f5c-9c1e-44ec-929a-28329eee74b7">

``` python
y_test_encoded = label_encoder.transform(y_test)
y_test_onehot = np_utils.to_categorical(y_test_encoded, num_classes)
test_loss, test_acc = model_DL.evaluate(X_test, y_test_onehot)
print('Test accuracy:', test_acc)
```

<div class="output stream stdout">

    83/83 [==============================] - 0s 2ms/step - loss: 0.9444 - accuracy: 0.5443
    Test accuracy: 0.5443471670150757

</div>

</div>
