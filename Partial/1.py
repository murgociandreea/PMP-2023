#1
from pgmpy.models import BayesianNetwork  # Actualizare conform avertizării
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


# Definirea structurii rețelei Bayesiene
model = BayesianNetwork([('StartMoneda', 'FirstRound'), ('StartMoneda', 'SecondRound'), ('FirstRound', 'SecondRound')])

# CPD pentru 'StartMoneda'
cpd_start = TabularCPD(variable='StartMoneda', variable_card=2, values=[[0.5], [0.5]],
                       state_names={'StartMoneda': ['P0', 'P1']})
# CPD pentru 'FirstRound'
cpd_first_round = TabularCPD(variable='FirstRound', variable_card=2,
                             values=[[0.6667, 0.5], [0.3333, 0.5]],
                             evidence=['StartMoneda'], evidence_card=[2],
                             state_names={'FirstRound': ['0_steme', '1_stema'], 'Start': ['P0', 'P1']})
#CPD pentru 'SecondRound'
#2*4
cpd_second_round = TabularCPD(variable='SecondRound', variable_card=2,
                              values=[[0.6667, 0.5, 0.5, 0.6667], [0.333, 0.5, 0.5, 0.333]],
                              evidence=['FirstRound', 'Start'], evidence_card=[2, 2],
                              state_names={'SecondRound': ['0_stema','1_stema', '2_stema', '3_stema'],
                                           'FirstRound': ['0_steme', '1_stema'],
                                           'Start': ['P0', 'P1']})


# Adăugarea CPD-urilor la model
model.add_cpds(cpd_start, cpd_first_round, cpd_second_round)

# Verificarea modelului
assert model.check_model()

# Crearea unui obiect de inferență
inference = VariableElimination(model)

# Calcularea probabilității pentru variabila FirstRound, având observația că în 'SecondRound' nu s-a obținut nicio stemă
result = inference.query(variables=['FirstRound'], evidence={'SecondRound': '0_stema'})

# Afișarea rezultatului
print(result)
