# 🚀 Guia de Execução do Projeto

Este documento descreve o fluxo padrão para configurar e executar o projeto em ambiente local.

O objetivo é garantir:

- Isolamento de dependências  
- Reprodutibilidade  
- Padronização do ambiente  
- Redução de conflitos entre versões  

---

# 📋 Pré-requisitos

Antes de iniciar, verifique se você possui:

- Python 3.9+ instalado
- Permissão para executar scripts no PowerShell (Windows)

Para verificar sua versão do Python:

```bash
python --version
```

---

# 🏗️ Setup do Ambiente

## 1️⃣ Criar o ambiente virtual

No diretório raiz do projeto, execute:

```bash
python -m venv .hacka_venv_prod
```

Isso criará um ambiente isolado chamado:

```
.hacka_venv_prod
```

---

## 2️⃣ Ativar o ambiente virtual

### Windows (PowerShell)

```powershell
.\.hacka_venv_prod\Scripts\Activate.ps1
```

Se a ativação for bem-sucedida, o terminal exibirá:

```
(.hacka_venv_prod)
```

---

## 3️⃣ Instalar as dependências

Com o ambiente ativado, instale os pacotes do projeto:

```bash
pip install -r requirements.txt
```

Isso instalará todas as bibliotecas necessárias descritas no arquivo `requirements.txt`.

---

## ▶️ Execução do Projeto

Após concluir a instalação das dependências, execute o script principal:

```bash
python 01_production.py
```

---

# 🔁 Fluxo Resumido

```bash
python -m venv .hacka_venv_prod
.\.hacka_venv_prod\Scripts\Activate.ps1
pip install -r requirements.txt
python 01_production.py
```

---

# 📌 Boas Práticas

- Nunca instale dependências globalmente.
- Sempre ative a venv antes de rodar o projeto.
- Não altere o `requirements.txt` sem atualizar o versionamento.
- Caso ocorram erros, recrie o ambiente virtual.

Para recriar o ambiente:

```bash
Remove-Item -Recurse -Force .hacka_venv_prod
python -m venv .hacka_venv_prod
```

---

# 🧠 Observação Importante

Comandos como:

```
!pip install ...
```

funcionam apenas em notebooks (ex: Jupyter).  
Eles **não funcionam em arquivos `.py` executados via terminal**.

---

# 🏁 Resultado Esperado

Após seguir o fluxo corretamente:

- Ambiente isolado criado
- Dependências instaladas
- Script executado sem erro de importação

Ambiente limpo. Processo replicável. Execução previsível.