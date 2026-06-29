# Juros compostos

Juros compostos são os juros calculados sobre o capital inicial e também sobre os
juros acumulados em períodos anteriores. É o efeito de "juros sobre juros", que
faz um investimento crescer de forma exponencial ao longo do tempo.

## A fórmula

Para um capital único, o valor futuro é `VF = P · (1 + i)^n`, onde `P` é o
principal, `i` é a taxa por período e `n` é o número de períodos. Quando há
aportes periódicos fixos, soma-se o valor futuro de uma série (anuidade):
`VF = P · (1 + i)^n + A · [((1 + i)^n − 1) / i]`, com `A` igual ao aporte por
período.

## Por que o tempo importa

Quanto maior o prazo, mais relevante fica a parcela de juros sobre juros. Começar
a investir cedo, mesmo com valores menores, costuma render mais do que começar
tarde com valores maiores, porque o tempo multiplica o efeito composto.

## Taxa nominal e aportes mensais

Uma taxa anual de 12% equivale, de forma aproximada, a 1% ao mês (12% / 12). Ao
projetar um investimento com aportes mensais, use a taxa mensal e o número de
meses. A ferramenta de juros compostos deste assistente faz exatamente esse
cálculo: capital inicial composto mês a mês mais a série de aportes mensais.
