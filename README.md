**Case:** Apply simple Logical regression with Pandas and visualize the result with matplotlib on dataset of Hiring decisions so that on parameters about the candidate the model can predict if next candidate would be hired or not.

**Libraries:** pandas, sklearn

**IMPORTING THE DATA:**
```python
df = pd.read_csv('recruitment_data.csv')
```
⤷ loads .csv file into a variable

```python
data.info()
```
⤷ returns sum of the datatypes in column
![[Pasted image 20240703234208.png]]

```python
X = data.drop('HiringDecision', axis=1)
```
⤷ creates new dataFrame from **data**, droping column (axis=1) where 'HiringDecision' persists

```python
INPUT_train, INPUT_test, OUT_train, OUT_test  = train_test_split(INPUT, OUT, train_size=0.85, random_state=13)
```
⤷ splits INPUT and OUTPUT values into training and testing dataset with random seed 13 (Pseudo-random)

**PREPROCESSING:**



**TRAINING THE MODEL:**
```python
model = LogisticRegression(max_iter=1200, random_state=13)
model.fit(INPUT_train, OUT_train)
```
⤷ create LogisticRegression model 
⤷ train it on training dataset arrays

```python
OUT_predict = model.predict(INPUT_test)
acc = accuracy_score(OUT_test, OUT_predict)
```
⤷ predict values from INPUT_test values
⤷ compare those values to values that fit

**V-I-S-U-A-L-I-Z-A-T-I-O-N:**

**Libraries:** pandas, seaborn, matplotlib

```python
data.hist(figsize=(12, 10))
plt.show()
```
⤷ Makes bargraph (histogram) for each column 
⤷ figsize defines each figure in **Inches**
![[Pasted image 20240704215938.png]]

```python
plt.figure(figsize=(32, 6))
sns.scatterplot(x='Experience', y='Score', data=data, hue='Decision')
plt.title('Experience vs Score')
plt.show()
```
⤷ make a figure <span style="color:rgb(142, 63, 202)">32</span> inches wide and <span style="color:rgb(142, 63, 202)">6</span> inches tall
⤷ the figure is <span style="color:rgb(255, 255, 0)">scatterplot</span> where each point represents record with x being Experience column, y Score column and color represents different decision
(color palette can be changed with **attribute palette** set to bright/pastel/deep/muted/dark/colorblind)
⤷ name figure <span style="color:rgb(0, 176, 80)">"Experience vs Score"</span> 
![[Pasted image 20240704222612.png]]