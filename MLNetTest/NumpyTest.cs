using Numpy;
using Xunit;
using Xunit.Abstractions;

namespace MLNetTest
{
    public class NumpyTest : AbstractUnitTest
    {
        public NumpyTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void Create()
        {
            var res = np.zeros(4, 3);
            Print(res);
            res[0] = np.ones(3);
            Print(res);
        }

        [Fact]
        public void Solve()
        {
            var B = np.array(new double[,] {{1, 3, 5}, {7, 9, 11}, {13, 15, 16}});
            var C = np.array(new double[,] {{54, 32}, {66, 11}, {75, 33}});
            var res = np.linalg.solve(B, C);
            Print(res);
        }
    }
}