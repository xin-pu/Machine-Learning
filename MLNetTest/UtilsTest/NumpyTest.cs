using System.Linq;
using Numpy;
using Xunit;
using Xunit.Abstractions;

namespace MLNetTest.UtilsTest
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

            var d = np.matmul(B, res);
            Print(d);
        }

        [Fact]
        public void Poly()
        {
            var x = np.ones(3, 1) * 2;
            Print(x);

            var npX = np.ones(x.shape[0], 3);
            Enumerable.Range(0, 3).ToList().ForEach(d =>
            {
                var row = np.ones(x.shape[0]) * d;
                npX[d] = np.power(x[0], row);
            });
            Print(npX);
            var y = np.transpose(npX);
            Print(y);
        }
    }
}