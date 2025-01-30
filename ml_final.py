# %%
import pandas as pd
import matplotlib as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("train.csv")
df2 = pd.read_csv("test.csv")

# %%
df.head(10)

# %%
df.shape

# %%
df.describe()

# %%
df.info()

# %%
df = df.drop(columns=["id", "Name", "City", "Profession", "Degree"
                      ])

# %%
df2 = df2.drop(columns=["id", "Name", "City","Profession", "Degree"])

# %%
df.isnull().sum()

# %%
df["Academic Pressure"].value_counts()

# %%
df["Dietary Habits"].value_counts()

# %%
df["Dietary Habits"] = df["Dietary Habits"].fillna("Moderate")

# %%
df2["Dietary Habits"].value_counts()

# %%
df2["Dietary Habits"] = df2["Dietary Habits"].fillna("Moderate")

# %%
die_cond = df["Dietary Habits"].value_counts()
df["Dietary Habits"] = df["Dietary Habits"].apply(lambda x: x if die_cond[x] > 20 else 'Moderate')

# %%
die_cond2 = df2["Dietary Habits"].value_counts()
df2["Dietary Habits"] = df2["Dietary Habits"].apply(lambda x: x if die_cond2[x] > 20 else 'Moderate')

# %%
df["Financial Stress"].value_counts()
df["Financial Stress"] = df["Financial Stress"].fillna(2)

# %%
df2["Financial Stress"] = df2["Financial Stress"].fillna(2)

# %%
sleep_cond = df["Sleep Duration"].value_counts()
df["Sleep Duration"] = df["Sleep Duration"].apply(lambda x: x if sleep_cond[x] > 100 else "Less than 5 hours")

# %%
sleep_cond2 = df2["Sleep Duration"].value_counts()
df2["Sleep Duration"] = df2["Sleep Duration"].apply(lambda x: x if sleep_cond2[x] > 100 else "Less than 5 hours")

# %%
df_stu = df.drop(["Work Pressure", "Job Satisfaction"], axis=1)

# %%
df_stu.isnull().sum()

# %%
df_stu = df_stu.dropna()

# %%
df_stu["Depression"].sum() / len(df_stu)

# %%
df_wk = df.drop(columns=["Academic Pressure", "CGPA", "Study Satisfaction"])

# %%
df_wk = df_wk.dropna()

# %%
df_wk["Depression"].sum() / len(df_wk)

# %%
### Graficos de idade

plt.figure(figsize=(8, 6))
sns.boxplot(data=df_stu, x='Depression', y='Age', palette=['salmon', 'skyblue'])

plt.title('Distribuição de Idade por Depressão da Base total')
plt.xlabel('Depressão')
plt.ylabel('Idade')
plt.xticks(ticks=[0, 1], labels=['Não', 'Sim'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_wk, x='Depression', y='Age', palette=['salmon', 'skyblue'])

plt.title('Distribuição de Idade por Depressão dos Trabalhadores')
plt.xlabel('Depressão')
plt.ylabel('Idade')
plt.xticks(ticks=[0, 1], labels=['Não', 'Sim'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_stu, x='Depression', y='Age', palette=['salmon', 'skyblue'])

plt.title('Distribuição de Idade por Depressão dos Estudantes')
plt.xlabel('Depressão')
plt.ylabel('Idade')
plt.xticks(ticks=[0, 1], labels=['Não', 'Sim'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.boxplot(data=df, x='Depression', y='Age', palette=['salmon', 'skyblue'], ax=axes[0])
axes[0].set_title('Distribuição de Idade por Depressão (Total)')
axes[0].set_xlabel('Depressão')
axes[0].set_ylabel('Idade')
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Não', 'Sim'])
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

sns.boxplot(data=df_wk, x='Depression', y='Age', palette=['salmon', 'skyblue'], ax=axes[1])
axes[1].set_title('Distribuição de Idade por Depressão (Trabalhadores)')
axes[1].set_xlabel('Depressão')
axes[1].set_ylabel('Idade')
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['Não', 'Sim'])
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

sns.boxplot(data=df_stu, x='Depression', y='Age', palette=['salmon', 'skyblue'], ax=axes[2])
axes[2].set_title('Distribuição de Idade por Depressão (Estudantes)')
axes[2].set_xlabel('Depressão')
axes[2].set_ylabel('Idade')
axes[2].set_xticks([0, 1])
axes[2].set_xticklabels(['Não', 'Sim'])
axes[2].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# %%

contagem = df['Depression'].value_counts()

plt.figure(figsize=(8, 5)) 

ax = contagem.plot(kind='bar', color=['salmon', 'skyblue'])

plt.title('Total de Pessoas com Depressão')
plt.xlabel('Depressão')
plt.ylabel('Total de Pessoas')
plt.xticks(rotation=0, ticks=[0, 1], labels=['Não', 'Sim'])

for i, v in enumerate(contagem):
    plt.text(i, v + 0.5, str(v), ha='center', fontsize=12)

plt.show()

# %%
contagem = df_stu['Depression'].value_counts()
print(contagem)
# %%
plt.figure(figsize=(8, 5)) 

ax = contagem.plot(kind='bar', color=['skyblue', 'salmon'])

plt.title('Total de Estudantes com Depressão')
plt.xlabel('Depressão')
plt.ylabel('Total de Pessoas')
plt.xticks(rotation=0, ticks=[1, 0], labels=['Não', 'Sim'])

for i, v in enumerate(contagem):
    plt.text(i, v + 0.5, str(v), ha='center', fontsize=12)

plt.show()

# %%
contagem = df_wk['Depression'].value_counts()

plt.figure(figsize=(8, 5)) 

ax = contagem.plot(kind='bar', color=['salmon', 'skyblue'])
plt.title('Total de Trabalhadores com Depressão')
plt.xlabel('Depressão')
plt.ylabel('Total de Pessoas')
plt.xticks(rotation=0, ticks=[0, 1], labels=['Não', 'Sim'])

for i, v in enumerate(contagem):
    plt.text(i, v + 0.5, str(v), ha='center', fontsize=12)

plt.show()

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

contagem = df['Depression'].value_counts()
contagem.plot(kind='bar', color=['salmon', 'skyblue'], ax=axes[0])
axes[0].set_title('Total de Pessoas com Depressão')
axes[0].set_xlabel('Depressão')
axes[0].set_ylabel('Total de Pessoas')
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Não', 'Sim'])

for i, v in enumerate(contagem):
    axes[0].text(i, v + 0.5, str(v), ha='center', fontsize=12)

contagem = df_stu['Depression'].value_counts()
contagem.plot(kind='bar', color=['skyblue', 'salmon'], ax=axes[1])
axes[1].set_title('Total de Estudantes com Depressão')
axes[1].set_xlabel('Depressão')
axes[1].set_ylabel('Total de Pessoas')
axes[1].set_xticks([1, 0])
axes[1].set_xticklabels(['Não', 'Sim'])

for i, v in enumerate(contagem):
    axes[1].text(i, v + 0.5, str(v), ha='center', fontsize=12)

contagem = df_wk['Depression'].value_counts()
contagem.plot(kind='bar', color=['salmon', 'skyblue'], ax=axes[2])
axes[2].set_title('Total de Trabalhadores com Depressão')
axes[2].set_xlabel('Depressão')
axes[2].set_ylabel('Total de Pessoas')
axes[2].set_xticks([0, 1])
axes[2].set_xticklabels(['Não', 'Sim'])

for i, v in enumerate(contagem):
    axes[2].text(i, v + 0.5, str(v), ha='center', fontsize=12)

plt.tight_layout()
plt.show()


# %%
# grafico por genero

contagem = df_stu.groupby(['Gender', 'Depression']).size().unstack()

ax = contagem.plot(kind='bar', figsize=(8,6), color=['red', 'blue'])

for p in ax.patches:
    ax.annotate(str(int(p.get_height())), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title('Quantidade da base total com e sem Depressão por Gênero')
plt.xlabel('Gênero')
plt.ylabel('Quantidade')
plt.legend(title='Depressão', labels=['Não', 'Sim'])
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

# %%
contagem = df.groupby(['Gender', 'Depression']).size().unstack()

ax = contagem.plot(kind='bar', figsize=(8,6), color=['red', 'blue'])

for p in ax.patches:
    ax.annotate(str(int(p.get_height())), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title('Quantidade de Estudantes com e sem Depressão por Gênero')
plt.xlabel('Gênero')
plt.ylabel('Quantidade')
plt.legend(title='Depressão', labels=['Não', 'Sim'])
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

# %%
contagem = df_wk.groupby(['Gender', 'Depression']).size().unstack()

ax = contagem.plot(kind='bar', figsize=(8,6), color=['red', 'blue'])

for p in ax.patches:
    ax.annotate(str(int(p.get_height())), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title('Quantidade de Trabalhadores com e sem Depressão por Gênero')
plt.xlabel('Gênero')
plt.ylabel('Quantidade')
plt.legend(title='Depressão', labels=['Não', 'Sim'])
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

cores = ['salmon', 'skyblue']

contagem = df.groupby(['Gender', 'Depression']).size().unstack()
contagem.plot(kind='bar', color=cores, ax=axes[0])

axes[0].set_title('Base Total: Depressão por Gênero')
axes[0].set_xlabel('Gênero')
axes[0].set_ylabel('Quantidade')
axes[0].legend(title='Depressão', labels=['Não', 'Sim'])
axes[0].set_xticks(range(len(contagem.index)))
axes[0].set_xticklabels(contagem.index, rotation=0)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

for p in axes[0].patches:
    axes[0].annotate(str(int(p.get_height())), 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='bottom', fontsize=12, fontweight='bold')

contagem = df_stu.groupby(['Gender', 'Depression']).size().unstack()
contagem.plot(kind='bar', color=cores, ax=axes[1])

axes[1].set_title('Estudantes: Depressão por Gênero')
axes[1].set_xlabel('Gênero')
axes[1].set_ylabel('Quantidade')
axes[1].legend(title='Depressão', labels=['Não', 'Sim'])
axes[1].set_xticks(range(len(contagem.index)))
axes[1].set_xticklabels(contagem.index, rotation=0)
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

for p in axes[1].patches:
    axes[1].annotate(str(int(p.get_height())), 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='bottom', fontsize=12, fontweight='bold')

contagem = df_wk.groupby(['Gender', 'Depression']).size().unstack()
contagem.plot(kind='bar', color=cores, ax=axes[2])

axes[2].set_title('Trabalhadores: Depressão por Gênero')
axes[2].set_xlabel('Gênero')
axes[2].set_ylabel('Quantidade')
axes[2].legend(title='Depressão', labels=['Não', 'Sim'])
axes[2].set_xticks(range(len(contagem.index)))
axes[2].set_xticklabels(contagem.index, rotation=0)
axes[2].grid(axis='y', linestyle='--', alpha=0.7)

for p in axes[2].patches:
    axes[2].annotate(str(int(p.get_height())), 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()

plt.show()

# %%
# grafico de estudantes com pressão academica
contagem = df_stu.groupby(['Academic Pressure', 'Depression']).size().unstack(fill_value=0)

ax = contagem.plot(kind='bar', figsize=(8,6), color=['red', 'blue'])

for p in ax.patches:
    ax.annotate(str(int(p.get_height())), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title('Quantidade de Estudantes com Depressão por Nível de Pressão (1 a 5)', fontsize=14)
plt.xlabel('Pressão da Faculdade (Escala de 1 a 5)', fontsize=12)
plt.ylabel('Quantidade de Pessoas', fontsize=12)
plt.legend(title='Depressão', labels=['Não', 'Sim'])
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

# %%
contagem = df_wk.groupby(['Work Pressure', 'Depression']).size().unstack(fill_value=0)

ax = contagem.plot(kind='bar', figsize=(8,6), color=['red', 'blue'])

for p in ax.patches:
    ax.annotate(str(int(p.get_height())), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title('Quantidade de Pessoas com Depressão por Nível de Pressão (1 a 5)', fontsize=14)
plt.xlabel('Pressão do Trabalho (Escala de 1 a 5)', fontsize=12)
plt.ylabel('Quantidade de Pessoas', fontsize=12)
plt.legend(title='Depressão', labels=['Não', 'Sim'])
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cores = ['salmon', 'skyblue']

contagem_stu = df_stu.groupby(['Academic Pressure', 'Depression']).size().unstack(fill_value=0)
contagem_stu.plot(kind='bar', color=cores, ax=axes[0])

axes[0].set_title('Estudantes: Depressão por Pressão Acadêmica (1 a 5)', fontsize=14)
axes[0].set_xlabel('Pressão Acadêmica (Escala de 1 a 5)', fontsize=12)
axes[0].set_ylabel('Quantidade de Pessoas', fontsize=12)
axes[0].legend(title='Depressão', labels=['Não', 'Sim'])
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

for p in axes[0].patches:
    axes[0].annotate(str(int(p.get_height())), 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='bottom', fontsize=12, fontweight='bold')

contagem_wk = df_wk.groupby(['Work Pressure', 'Depression']).size().unstack(fill_value=0)
contagem_wk.plot(kind='bar', color=cores, ax=axes[1])

axes[1].set_title('Trabalhadores: Depressão por Pressão no Trabalho (1 a 5)', fontsize=14)
axes[1].set_xlabel('Pressão no Trabalho (Escala de 1 a 5)', fontsize=12)
axes[1].set_ylabel('Quantidade de Pessoas', fontsize=12)
axes[1].legend(title='Depressão', labels=['Não', 'Sim'])
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

for p in axes[1].patches:
    axes[1].annotate(str(int(p.get_height())), 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()

plt.show()

# %%
# grafico por duração de sono
contagem = df_stu.groupby(['Sleep Duration', 'Depression']).size().unstack(fill_value=0)

ax = contagem.plot(kind='barh', figsize=(8,6), color=['#FF7F7F', "#ADD8E6"])

for p in ax.patches:
    ax.annotate(str(int(p.get_width())), 
                (p.get_x() + p.get_width() / 2., p.get_y() + p.get_height() / 2), 
                ha='center', va='center', fontsize=12, fontweight='bold')

plt.title('Quantidade de Estudantes com e sem Depressão por Duração do Sono', fontsize=14)
plt.xlabel('Quantidade de Estudantes', fontsize=12)
plt.ylabel('Duração do Sono', fontsize=12)
plt.legend(title='Depressão', labels=["Não",'Sim'])

plt.show()

# %%
contagem = df_wk.groupby(['Sleep Duration', 'Depression']).size().unstack(fill_value=0)

ax = contagem.plot(kind='barh', figsize=(8,6), color=['#FF7F7F', "#ADD8E6"])

for p in ax.patches:
    ax.annotate(str(int(p.get_width())), 
                (p.get_x() + p.get_width() / 2., p.get_y() + p.get_height() / 2), 
                ha='center', va='center', fontsize=12, fontweight='bold')

plt.title('Quantidade de Pessoas com e sem Depressão por Duração do Sono', fontsize=14)
plt.xlabel('Quantidade de Pessoas', fontsize=12)
plt.ylabel('Duração do Sono', fontsize=12)
plt.legend(title='Depressão', labels=["Não",'Sim'])

plt.show()

# %%
contagem = df.groupby(['Sleep Duration', 'Depression']).size().unstack(fill_value=0)

ax = contagem.plot(kind='barh', figsize=(8,6), color=['#FF7F7F', "#ADD8E6"])

for p in ax.patches:
    ax.annotate(str(int(p.get_width())), 
                (p.get_x() + p.get_width() / 2., p.get_y() + p.get_height() / 2), 
                ha='center', va='center', fontsize=12, fontweight='bold')

plt.title('Quantidade de Pessoas com e sem Depressão por Duração do Sono', fontsize=14)
plt.xlabel('Quantidade de Pessoas', fontsize=12)
plt.ylabel('Duração do Sono', fontsize=12)
plt.legend(title='Depressão', labels=["Não",'Sim'])

plt.show()

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

cores = ['#FF7F7F', '#ADD8E6']

contagem = df_stu.groupby(['Sleep Duration', 'Depression']).size().unstack(fill_value=0)
contagem.plot(kind='barh', color=cores, ax=axes[0])

axes[0].set_title('Estudantes: Depressão por Duração do Sono', fontsize=14)
axes[0].set_xlabel('Quantidade de Estudantes', fontsize=12)
axes[0].set_ylabel('Duração do Sono', fontsize=12)
axes[0].legend(title='Depressão', labels=["Não", 'Sim'])

for p in axes[0].patches:
    axes[0].annotate(str(int(p.get_width())), 
                     (p.get_x() + p.get_width() / 2., p.get_y() + p.get_height() / 2), 
                     ha='center', va='center', fontsize=12, fontweight='bold')

contagem = df_wk.groupby(['Sleep Duration', 'Depression']).size().unstack(fill_value=0)
contagem.plot(kind='barh', color=cores, ax=axes[1])

axes[1].set_title('Trabalhadores: Depressão por Duração do Sono', fontsize=14)
axes[1].set_xlabel('Quantidade de Pessoas', fontsize=12)
axes[1].set_ylabel('Duração do Sono', fontsize=12)
axes[1].legend(title='Depressão', labels=["Não", 'Sim'])

for p in axes[1].patches:
    axes[1].annotate(str(int(p.get_width())), 
                     (p.get_x() + p.get_width() / 2., p.get_y() + p.get_height() / 2), 
                     ha='center', va='center', fontsize=12, fontweight='bold')

contagem = df.groupby(['Sleep Duration', 'Depression']).size().unstack(fill_value=0)
contagem.plot(kind='barh', color=cores, ax=axes[2])

axes[2].set_title('Base Total: Depressão por Duração do Sono', fontsize=14)
axes[2].set_xlabel('Quantidade de Pessoas', fontsize=12)
axes[2].set_ylabel('Duração do Sono', fontsize=12)
axes[2].legend(title='Depressão', labels=["Não", 'Sim'])

for p in axes[2].patches:
    axes[2].annotate(str(int(p.get_width())), 
                     (p.get_x() + p.get_width() / 2., p.get_y() + p.get_height() / 2), 
                     ha='center', va='center', fontsize=12, fontweight='bold')

plt.tight_layout()

plt.show()

# %%
# grafico sobre Hábitos alimentares
contagem = df_stu.groupby(['Dietary Habits', 'Depression']).size().unstack(fill_value=0)

ax = contagem.plot(kind='barh', figsize=(8,6), color=['salmon','skyblue'])

for p in ax.patches:
    ax.annotate(str(int(p.get_width())), 
                (p.get_x() + p.get_width() / 2., p.get_y() + p.get_height() / 2), 
                ha='center', va='center', fontsize=12, fontweight='bold')

plt.title('Estudantes com Depressão de acordo com a Alimentação', fontsize=14)
plt.xlabel('Quantidade de Pessoas', fontsize=12)
plt.ylabel('Duração do Sono', fontsize=12)
plt.legend(title='Depressão', labels=["Não",'Sim'])

plt.show()

# %%
contagem = df.groupby(['Dietary Habits', 'Depression']).size().unstack(fill_value=0)

ax = contagem.plot(kind='barh', figsize=(8,6), color=['salmon','skyblue'])

for p in ax.patches:
    ax.annotate(str(int(p.get_width())), 
                (p.get_x() + p.get_width() / 2., p.get_y() + p.get_height() / 2), 
                ha='center', va='center', fontsize=12, fontweight='bold')

plt.title('Pessoas com Depressão de acordo com a Alimentação', fontsize=14)
plt.xlabel('Quantidade de Pessoas', fontsize=12)
plt.ylabel('Duração do Sono', fontsize=12)
plt.legend(title='Depressão', labels=["Não",'Sim'])

plt.show()

# %%
contagem = df_wk.groupby(['Dietary Habits', 'Depression']).size().unstack(fill_value=0)

ax = contagem.plot(kind='barh', figsize=(8,6), color=['salmon','skyblue'])

for p in ax.patches:
    ax.annotate(str(int(p.get_width())), 
                (p.get_x() + p.get_width() / 2., p.get_y() + p.get_height() / 2), 
                ha='center', va='center', fontsize=12, fontweight='bold')

plt.title('Trabalhadores com Depressão de acordo com a Alimentação', fontsize=14)
plt.xlabel('Quantidade de Pessoas', fontsize=12)
plt.ylabel('Duração do Sono', fontsize=12)
plt.legend(title='Depressão', labels=["Não",'Sim'])

plt.show()

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

cores = ['salmon', 'skyblue']

contagem = df_stu.groupby(['Dietary Habits', 'Depression']).size().unstack(fill_value=0)
contagem.plot(kind='barh', color=cores, ax=axes[0])

axes[0].set_title('Estudantes: Depressão por Alimentação', fontsize=14)
axes[0].set_xlabel('Quantidade de Pessoas', fontsize=12)
axes[0].set_ylabel('Hábitos Alimentares', fontsize=12)
axes[0].legend(title='Depressão', labels=["Não", 'Sim'])

for p in axes[0].patches:
    axes[0].annotate(str(int(p.get_width())), 
                     (p.get_x() + p.get_width() / 2., p.get_y() + p.get_height() / 2), 
                     ha='center', va='center', fontsize=12, fontweight='bold')

contagem = df.groupby(['Dietary Habits', 'Depression']).size().unstack(fill_value=0)
contagem.plot(kind='barh', color=cores, ax=axes[1])

axes[1].set_title('Base Total: Depressão por Alimentação', fontsize=14)
axes[1].set_xlabel('Quantidade de Pessoas', fontsize=12)
axes[1].set_ylabel('Hábitos Alimentares', fontsize=12)
axes[1].legend(title='Depressão', labels=["Não", 'Sim'])

for p in axes[1].patches:
    axes[1].annotate(str(int(p.get_width())), 
                     (p.get_x() + p.get_width() / 2., p.get_y() + p.get_height() / 2), 
                     ha='center', va='center', fontsize=12, fontweight='bold')

contagem = df_wk.groupby(['Dietary Habits', 'Depression']).size().unstack(fill_value=0)
contagem.plot(kind='barh', color=cores, ax=axes[2])

axes[2].set_title('Trabalhadores: Depressão por Alimentação', fontsize=14)
axes[2].set_xlabel('Quantidade de Pessoas', fontsize=12)
axes[2].set_ylabel('Hábitos Alimentares', fontsize=12)
axes[2].legend(title='Depressão', labels=["Não", 'Sim'])

for p in axes[2].patches:
    axes[2].annotate(str(int(p.get_width())), 
                     (p.get_x() + p.get_width() / 2., p.get_y() + p.get_height() / 2), 
                     ha='center', va='center', fontsize=12, fontweight='bold')

plt.tight_layout()

plt.show()

# %%
df["Academic Pressure"] = df["Academic Pressure"].fillna(0)
df["Work Pressure"] = df["Work Pressure"].fillna(0)
df["CGPA"] = df["CGPA"].fillna(0)
df["Job Satisfaction"] = df["Job Satisfaction"].fillna(0)
df["Study Satisfaction"] = df["Study Satisfaction"].fillna(0)

# %%
df2["Academic Pressure"] = df2["Academic Pressure"].fillna(0)
df2["Work Pressure"] = df2["Work Pressure"].fillna(0)
df2["CGPA"] = df2["CGPA"].fillna(0)
df2["Job Satisfaction"] = df2["Job Satisfaction"].fillna(0)
df2["Study Satisfaction"] = df2["Study Satisfaction"].fillna(0)

# %%
df_num = df.select_dtypes(include=['number'])

# %%
df_corr = df_num.corr()
df_corr

# %%
mask = np.triu(np.ones_like(df_corr, dtype=bool))
plt.figure(figsize=(12, 8))
sns.heatmap(df_corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.show()

# %%
contagem = df_stu.groupby(['Gender', 'Depression']).size().unstack()

contagem_feminino = contagem.loc['Female']
contagem_masculino = contagem.loc['Male']

fig, ax = plt.subplots(figsize=(8, 6))
bar_width = 0.35  
index = range(len(contagem_feminino)) 

bar1 = ax.bar([i - bar_width/2 for i in index], contagem_feminino, bar_width, color='skyblue', label='Feminino')
bar2 = ax.bar([i + bar_width/2 for i in index], contagem_masculino, bar_width, color='salmon', label='Masculino')

for bar in bar1 + bar2:
    altura = bar.get_height()
    ax.annotate(f'{altura}', 
                xy=(bar.get_x() + bar.get_width() / 2, altura), 
                xytext=(0, 3), 
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

ax.set_title('Depressão por Sexo')
ax.set_xlabel('Depressão')
ax.set_ylabel('Quantidade')
ax.set_xticks(index)
ax.set_xticklabels(['Tem Depressão', 'Não Tem Depressão'])
ax.legend()

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %%
from sklearn.preprocessing import OneHotEncoder

# %%
onehot = OneHotEncoder(sparse_output=False, drop='first') 

# %%
onehot_encoded = onehot.fit_transform(df[['Gender',
                                          "Sleep Duration", "Dietary Habits",
                                          "Have you ever had suicidal thoughts ?",
                                          "Family History of Mental Illness"]])

# %%
onehot_encoded2 = onehot.fit_transform(df2[['Gender',
                                          "Sleep Duration", "Dietary Habits",
                                          "Have you ever had suicidal thoughts ?",
                                          "Family History of Mental Illness"]])

# %%
encoded_df = pd.DataFrame(
    onehot_encoded, columns=onehot.get_feature_names_out(['Gender',
                                          "Sleep Duration", "Dietary Habits", 
                                          "Have you ever had suicidal thoughts ?",
                                          "Family History of Mental Illness"]))

# %%
encoded_df2 = pd.DataFrame(
    onehot_encoded2, columns=onehot.get_feature_names_out(['Gender',
                                          "Sleep Duration", "Dietary Habits",
                                          "Have you ever had suicidal thoughts ?",
                                          "Family History of Mental Illness"]))

# %%
encoded_df

# %%
categorical_columns = df.select_dtypes(include="object")

# %%
categorical_columns2 = df2.select_dtypes(include="object")

# %%
df_final = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)

# %%
df_final2 = pd.concat([df2.drop(columns=categorical_columns2), encoded_df2], axis=1)

# %%
df_final["Depression"]

# %%
df_final

# %%
from sklearn.model_selection import train_test_split

# %%
X = df_final.drop('Depression', axis=1)
y = df_final['Depression']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.35, random_state=42)

# %%
X2 = df_final2

# %%
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# %%
gbc = GradientBoostingClassifier(random_state=42)

# %%
param_grid = {
    'n_estimators': [50, 100, 200],        
    'learning_rate': [0.01, 0.1, 0.2],    
    'max_depth': [3, 4, 5],               
    'subsample': [0.8, 1.0],              
    'min_samples_split': [2, 5, 10],    
}

grid_search = GridSearchCV(
    estimator=gbc,
    param_grid=param_grid,
    scoring='accuracy',          
    cv=3,                        
    verbose=2,                   
    n_jobs=-1                    
)

# %%
grid_search.fit(X_train, y_train)


# %%
best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(X_val)


# %%
importances = best_model.feature_importances_  
feature_names = X_train.columns  
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

importance_df = importance_df.sort_values(by='Importance', ascending=False)

# %%
importance_df

# %%
plt.figure(figsize=(10, 6))

sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')

plt.title('Importância das Features', fontsize=16)
plt.xlabel('Importância', fontsize=12)
plt.ylabel('Feature', fontsize=12)

plt.show()

# %%
test_predictions = best_model.predict(df_final2)

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score

# %%
accuracy = accuracy_score(y_val, y_val_pred)
print(f'Acurácia: {accuracy:.4f}')

f1 = f1_score(y_val, y_val_pred, average='binary')
print(f'F1-Score: {f1:.4f}')

# %%
cm = confusion_matrix(y_val, y_val_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Não', 'Sim'], yticklabels=['Não', 'Sim'])
plt.xlabel('Previsões')
plt.ylabel('Valores Reais')
plt.title('Matriz de Confusão')
plt.show()

# %%
df_id = pd.read_csv("test.csv")

# %%
output = pd.DataFrame({'id': df_id["id"], 'Depression': test_predictions})
output.to_csv('sample_submission.csv', index=False)

# %%
df_final2

# %%
