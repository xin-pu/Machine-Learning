using System.Linq;
using MLNet.Utils;
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
        public void create()
        {
            var zeros = np.zeros(4, 3);
            print(zeros);
            zeros[0] = np.ones(3);
            print(zeros);

            var random = np.random.rand(4, 3);
            print(random);
        }


        [Fact]
        public void load()
        {
            var ones = np.random.rand(2, 1);
            print(ones);
            print(ones.shape);
            var a = ones["0,:"].GetData<double>();
            print(a);
        }

        [Fact]
        public void multiply()
        {
            var x = np.array(new double[,] {{1, 2}, {1, 2}});
            var y = np.array(new double[,] {{2, 3}, {1, 2}});
            var res = np.matmul(x, y);
            print(res);
        }


        [Fact]
        public void power()
        {
            var x = np.array(new double[,] {{1, 2, 3}, {1, 2, 3}});
            var res = np.power(x, np.array(2));
            print(res);
        }


        [Fact]
        public void solve()
        {
            var B = np.array(new double[,] {{1, 3, 5}, {7, 9, 11}, {13, 15, 16}});
            var C = np.array(new double[,] {{54, 32}, {66, 11}, {75, 33}});
            var res = np.linalg.solve(B, C);
            print(res);

            var d = np.matmul(B, res);
            print(d);
        }


        [Fact]
        public void sum()
        {
            var a = np.random.rand(5, 2);
            print(a);
            var res = np.sum(a, 0);
            print(res);
        }

        [Fact]
        public void sigmoid()
        {
            var a = np.array(new[,] {{0.8, 0.1}, {1.2, 2}});
            print(a);
            var res = np2.sigmoid(a);
            print(res);
        }

        [Fact]
        public void norm()
        {
            var a = np.random.randn(2);
            print(a);
            print(np.linalg.norm(a, 2));
        }

        [Fact]
        public void shuffle()
        {
            var a = np.random.randn(5, 3);
            print(a);
            np.random.shuffle(a);

            print(a);
        }

        [Fact]
        public void index()
        {
            var a = np.random.randn(5, 3);
            print(a);
            var index = "1:3,:";
            var batch_x = a[index];
            print(batch_x);
        }

        [Fact]
        public void poly()
        {
            var x = np.ones(3, 1) * 2;
            print(x);

            var npX = np.ones(x.shape[0], 3);
            Enumerable.Range(0, 3).ToList().ForEach(d =>
            {
                var row = np.ones(x.shape[0]) * d;
                npX[d] = np.power(x[0], row);
            });
            print(npX);
            var y = np.transpose(npX);
            print(y);
        }


        [Fact]
        public void linear_first_order()
        {
            var a = np.array(new double[,] {{1, 1, 1}, {0, 0, 0}});
            var res = transformer.to_linear_firstorder(a);
            print(res);
        }

        [Fact]
        public void equal()
        {
            var a = np.array(1, 0, 1, 0);
            var b = np.array(1, 1, 0, 0);


            var where = np.equal(a, np.array(0));
            print(where);
            var res2 = np.bitwise_and(np.equal(a, b), where);
            print(res2);
        }
    }
}