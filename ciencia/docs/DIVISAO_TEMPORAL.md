# ⏰ IMPORTANTE: Divisão Temporal dos Dados

## 🎯 Estratégia de Divisão

Este projeto utiliza **divisão temporal por SAFRA** ao invés de divisão aleatória tradicional:

### 📅 Conjuntos de Dados:

**TREINO (69% - 873.297 registros):**
- SAFRA 202410 (Out/2024): 203.828 registros
- SAFRA 202411 (Nov/2024): 226.119 registros
- SAFRA 202412 (Dez/2024): 225.760 registros
- SAFRA 202501 (Jan/2025): 217.590 registros

**TESTE (31% - 398.798 registros):**
- SAFRA 202502 (Fev/2025): 198.069 registros
- SAFRA 202503 (Mar/2025): 200.729 registros

## ⚠️ Por que isso é CRUCIAL?

### 1. **Simula Produção Real**
Em produção, o modelo sempre faz predições em dados **futuros** (que não viu durante o treinamento).
A divisão temporal garante que estamos testando exatamente esse cenário.

### 2. **Evita Data Leakage**
Com divisão aleatória, podemos ter:
- Clientes do mesmo mês em treino e teste
- Informações futuras "vazando" para o treino
- Performance otimista e não realista

### 3. **Detecta Drift Temporal**
Se o modelo performar mal no teste temporal, indica que:
- Padrões mudaram ao longo do tempo
- Modelo pode não generalizar para o futuro
- Necessário retreinamento mais frequente

### 4. **Avaliação Honesta**
Performance no teste temporal é a **verdadeira** capacidade preditiva do modelo.

## 📊 Comparação

| Aspecto | Divisão Aleatória | Divisão Temporal |
|---------|-------------------|------------------|
| **Realismo** | ❌ Baixo | ✅ Alto |
| **Data Leakage** | ⚠️ Risco alto | ✅ Sem risco |
| **Performance** | 📈 Otimista | 📉 Realista |
| **Produção** | ❌ Não reflete | ✅ Reflete |

## 🔍 Exemplo Prático

**Divisão Aleatória (ERRADO para séries temporais):**
```
Treino:   [Out, Nov, Dez, Jan, Fev, Mar] - clientes misturados
Teste:    [Out, Nov, Dez, Jan, Fev, Mar] - clientes misturados
❌ Cliente de Março pode estar no treino!
```

**Divisão Temporal (CORRETO):**
```
Treino:   [Out, Nov, Dez, Jan] - todos os clientes desses meses
Teste:    [Fev, Mar]           - todos os clientes desses meses
✅ Modelo nunca viu nenhum cliente de Fev/Mar!
```

## 💡 Implicações para o Projeto

1. **Performance pode ser menor** que com divisão aleatória (mas é a **real**!)
2. **Modelo está realmente sendo testado** em cenário de produção
3. **Resultados são confiáveis** para tomada de decisão
4. **Retreinamento periódico** é necessário (drift temporal)

## 🎓 Boas Práticas

Para projetos de classificação com componente temporal:

✅ **SEMPRE use divisão temporal**
✅ **Teste em múltiplos períodos futuros** se possível
✅ **Monitore performance ao longo do tempo**
✅ **Retreine regularmente** (ex: mensalmente)

❌ **NUNCA use divisão aleatória** em dados com timestamp/safra
❌ **NUNCA** misture períodos futuros no treino

## 📚 Referências

- ["On the dangers of cross-validation"](https://www.sciencedirect.com/science/article/pii/S0925231217309864)
- ["Time Series Split"](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- Best practices em Machine Learning temporal

---

**Implementado em:** `02_tratamento_dados.ipynb`

**Validação:** Antes de usar qualquer modelo em produção, sempre verifique se a divisão temporal foi respeitada!
