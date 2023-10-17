from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

model = BayesianNetwork([('Cutremur', 'Incendiu'), ('Incendiu', 'Alarma'), ('Cutremur', 'Alarma')])

cpd_cutremur = TabularCPD(variable='Cutremur', variable_card=2, values=[[0.9995], [0.0005]])
cpd_incendiu = TabularCPD(variable='Incendiu', variable_card=2, values=[[0.99], [0.01]])
cpd_alarma = TabularCPD(variable='Alarma', variable_card=2,
                                   values=[[0.9999, 0.98, 0.05, 0.02],
                                           [0.0001, 0.02, 0.95, 0.98]],
                                   evidence=['Cutremur', 'Incendiu'], evidence_card=[2, 2])


model.add_cpds(cpd_cutremur, cpd_incendiu, cpd_alarma)

assert model.check_model()

infer = VariableElimination(model)
result = infer.query(variables=['Cutremur'], evidence={'Alarma': 1})
print(result)

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()