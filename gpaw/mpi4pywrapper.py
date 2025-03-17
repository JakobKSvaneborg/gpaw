import mpi4py.MPI as mpi


class MPI4PYWrapper:
    def __init__(self, comm, parent=None):
        self.comm = comm
        self.size = comm.size
        self.rank = comm.rank
        self.parent = parent  # XXX check C-object against comm.parent?

    def new_communicator(self, ranks):
        comm = self.comm.Create(self.comm.group.Incl(ranks))
        if self.comm.rank in ranks:
            return MPI4PYWrapper(comm, parent=self)
        else:
            # This cpu is not in the new communicator:
            return None

    def sum(self, a, root=-1, op=mpi.SUM):
        if isinstance(a, (int, float, complex)):
            if root == -1:
                return self.comm.allreduce(a, op=op)
            else:
                return self.comm.reduce(a, root=root, op=op)
        else:
            if root == -1:
                self.comm.Allreduce(a, a, op=op)
            else:
                self.comm.Reduce(a, a, root=root, op=op)

    def product(self, a, root=-1):
        return self.sum(a, root, mpi.PROD)

    def max(self, a, root=-1):
        return self.sum(a, root, mpi.MAX)

    def min(self, a, root=-1):
        return self.sum(a, root, mpi.MIN)

    def scatter(self, a, b, root):
        self.comm.Scatter(a, b, root)

    def alltoallv(self, sbuffer, scounts, sdispls, rbuffer, rcounts, rdispls):
        self.comm.Alltoallv((sbuffer, (scounts, sdispls), sbuffer.dtype.char),
                            (rbuffer, (rcounts, rdispls), rbuffer.dtype.char))

    def all_gather(self, a, b):
        self.comm.Allgather(a, b)

    def gather(self, a, root, b=None):
        self.comm.Gather(a, b, root)

    def broadcast(self, a, root):
        self.comm.Bcast(a, root)

    def sendreceive(self, a, dest, b, src, sendtag=123, recvtag=123):
        return self.comm.Sendrecv(a, dest, sendtag, b, src, recvtag)

    def send(self, a, dest, tag=123, block=True):
        if block:
            self.comm.Send(a, dest, tag)
        else:
            return self.comm.Isend(a, dest, tag)

    def ssend(self, a, dest, tag=123):
        return self.comm.Ssend(a, dest, tag)

    def receive(self, a, src, tag=123, block=True):
        if block:
            self.comm.Recv(a, src, tag)
        else:
            return self.comm.Irecv(a, src, tag)

    def test(self, request):
        return request.test()

    def testall(self, requests):
        return mpi.Request.testall(requests)

    def wait(self, request):
        request.wait()

    def waitall(self, requests):
        mpi.Request.waitall(requests)

    def abort(self, errcode):
        """Terminate MPI execution environment of all tasks in the group.
        This function only returns in the advent of an error occurring.

        Parameters:

        errcode: int
            Error code to return to the invoking environment.

        """
        1 / 0
        return self.comm.abort(errcode)

    def name(self):
        return self.comm.Get_name()

    def barrier(self):
        self.comm.barrier()

    def compare(self, othercomm):
        """Compare communicator to other.

        Returns 'ident' if they are identical, 'congruent' if they are
        copies of each other, 'similar' if they are permutations of
        each other, and otherwise 'unequal'.

        This method corresponds to MPI_Comm_compare."""
        1 / 0
        # if isinstance(self.comm, SerialCommunicator):
        #     return self.comm.compare(othercomm.comm)  # argh!
        result = self.comm.compare(othercomm.get_c_object())
        assert result in ['ident', 'congruent', 'similar', 'unequal']
        return result

    def translate_ranks(self, other, ranks):
        """"Translate ranks from communicator to other.

        ranks must be valid on this communicator.  Returns ranks
        on other communicator corresponding to the same processes.
        Ranks that are not defined on the other communicator are
        assigned values of -1.  (In contrast to MPI which would
        assign MPI_UNDEFINED)."""
        1 / 0
        assert hasattr(other, 'translate_ranks'), \
            'Excpected communicator, got %s' % other
        assert all(0 <= rank for rank in ranks)
        assert all(rank < self.size for rank in ranks)
        # if isinstance(self.comm, SerialCommunicator):
        #     return self.comm.translate_ranks(other.comm, ranks)  # argh!
        otherranks = self.comm.translate_ranks(other.get_c_object(), ranks)
        assert all(-1 <= rank for rank in otherranks)
        assert ranks.dtype == otherranks.dtype
        return otherranks

    def get_members(self):
        """Return the subset of processes which are members of this MPI group
        in terms of the ranks they are assigned on the parent communicator.
        For the world communicator, this is all integers up to ``size``.

        Example::

          >>> world.rank, world.size
          (3, 4)
          >>> world.get_members()
          array([0, 1, 2, 3])
          >>> comm = world.new_communicator(array([2, 3]))
          >>> comm.rank, comm.size
          (1, 2)
          >>> comm.get_members()
          array([2, 3])
          >>> comm.get_members()[comm.rank] == world.rank
          True

        """
        1 / 0
        return self.comm.get_members()

    def get_c_object(self):
        return self.comm


serial_comm = MPI4PYWrapper(mpi.COMM_SELF)
