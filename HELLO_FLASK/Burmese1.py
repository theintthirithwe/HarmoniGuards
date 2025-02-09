import pandas as pd
import numpy as np
import re
import string
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib


# In[2]:


file_path = r"C:\Users\USER\hatespeech - burmese\updated_burmese_hatespeech_and_normalspeech_dataset.csv"
df = pd.read_csv(file_path)


# In[ ]:


pd.set_option('display.max_rows', None)


# In[ ]:


# Step 3: Data Preprocessing
# Cleaning function for Burmese text
def clean(text):
    text = str(text).lower()
    text = re.sub(r'[^\u1000-\u109F ]', '', text)  # Keep only Burmese characters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text


# In[ ]:


# Apply cleaning to the dataset
df["Text"] = df["Text"].apply(clean)


# In[ ]:


# Prepare features and labels
x = np.array(df["Text"])
y = np.array(df["Label"])


# In[ ]:


# Vectorize the text data
cv = CountVectorizer(analyzer='char', ngram_range=(3, 5))  
x = cv.fit_transform(df["Text"])


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[ ]:


# Train a Random Forest Classifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
joblib.dump(clf, 'burmese_decision_tree_model_random.joblib')
joblib.dump(cv, "burmese_vectorizer_random.joblib")
print("\nModel and vectorizer saved successfully.")


# In[ ]:


# Evaluate the model
train_accuracy = rf_clf.score(X_train, y_train)
test_accuracy = rf_clf.score(X_test, y_test)


# In[ ]:


print("\nEvaluation:")
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")


# In[ ]:


# Generate a classification report
y_pred = rf_clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=1))


# In[ ]:


# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=["Hate Speech", "Normal Speech"], yticklabels=["Hate Speech", "Normal Speech"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix for Hate Speech Detection")
plt.show()


# In[ ]:


# Test with custom examples
def predict_text(text):
    cleaned_text = clean(text)
    text_vector = cv.transform([cleaned_text]).toarray()
    prediction = rf_clf.predict(text_vector)
    return prediction[0]


# In[ ]:


# Test cases
test_cases = [
    "ဒီဖေ", 
    "ကြက်ဥခွဲမယ်",  
    "အလုပ်ကိုလုပ်ရအောင်",
    "တစ်လုံးမှမပြောချင်ဘူး",
    "သေချာသုံး",
    "ကုလား",
    "Company",
    "တစ်လုံးမှမပြောချင်ဘူး",
    "ကုလားပဲ",
    "ကုလားပဲဟင်း စားမလား",
    "စောက်စကားမများနဲ့",
    "မင်္ဂလာပါ ထမင်းစားမလားဗျ",
    "မင်္ဂလာပါ",
    "လီးစကားတွေ လာပြောနေတာပဲ",
    "လိမ္မော်သီးစားမလား"
    
]


# In[ ]:


print("\nCustom Predictions:")
for case in test_cases:
    result = predict_text(case)
    print(f"Text: {case} → Prediction: {result}")


# In[ ]:




