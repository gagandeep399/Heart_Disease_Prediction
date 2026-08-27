# 🚀 GitHub par Push Karne ke Steps

## 1. GitHub par naya repo banao
- https://github.com/new par jao
- Repo name do (e.g. `heart-disease-prediction`)
- "Add README" checkbox **mat** check karo
- "Create repository" click karo

## 2. Repo ka URL copy karo
Kuch aisa dikhega:
```
https://github.com/<your-username>/heart-disease-prediction.git
```

## 3. Terminal me ye commands chalao (zip extract karne ke baad)

```bash
cd heart_disease_prediction

git init
git add .
git commit -m "Initial commit: Heart Disease Prediction ML project"
git branch -M main
git remote add origin https://github.com/<your-username>/heart-disease-prediction.git
git push -u origin main
```

> `<your-username>` aur repo name apne hisaab se badal dena.

## 4. Agar authentication maange
GitHub ab password nahi, **Personal Access Token (PAT)** maangta hai:
- https://github.com/settings/tokens par jaake naya token banao (scope: `repo`)
- Push karte waqt password ki jagah wahi token paste karo

Bas! Tumhara project GitHub par live ho jayega. ✅
