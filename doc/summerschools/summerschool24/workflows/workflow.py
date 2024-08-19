import taskblaster as tb


@tb.workflow
class MaterialsWorkflow:
    atoms = tb.var()
    calculator = tb.var()

    @tb.task
    def relax(self):
        return tb.node(
            'optimize_cell',
            atoms=self.atoms,
            calculator=self.calculator)

    @tb.task
    def groundstate(self):
        return tb.node(
            'groundstate', atoms=self.relax,
            calculator=self.calculator)

    @tb.task
    def bandstructure(self):
        return tb.node('bandstructure', gpw=self.groundstate)


def workflow(runner):
    from ase.build import bulk
    wf = MaterialsWorkflow(
        atoms=bulk('Si'),
        calculator={'mode': 'pw',
                    'kpts': (4, 4, 4),
                    'txt': 'gpaw.txt'})
    runner.run_workflow(wf)
# end-workflow-function-snippet
