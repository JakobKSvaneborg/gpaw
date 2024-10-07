import taskblaster as tb


@tb.workflow
class ParametrizedMaterialsWorkflow:
    calculator = tb.var()

    @tb.dynamical_workflow_generator({'symbols': '*/atoms'})
    def systems(self):
        return tb.node('parametrize_materials_workflow',
                       calculator=self.calculator)


def workflow(rn):
    calculator = {
        'mode': 'pw',
        'kpts': {'density': 1.0},
        'txt': 'gpaw.txt'
    }
    wf = ParametrizedMaterialsWorkflow(calculator=calculator)
    rn.run_workflow(wf)
