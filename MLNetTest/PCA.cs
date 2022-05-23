using Numpy;
using Xunit;
using Xunit.Abstractions;

namespace MLNetTest
{
    public class PCA : AbstractUnitTest
    {
        public PCA(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }


        /// <summary>
        ///     Based on eigenvalue decomposition
        /// </summary>
        [Fact]
        public void PCATest()
        {
            var X = np.array(new double[,] {{-1, -1, 0, 2, 0}, {-2, 0, 0, 1, 1}});
            print(X);
            var mean = np.mean(X, -1, keepdims: true);
            var std = X - mean;
            print(mean);
            print(std);
            var feature = X.shape[0];
            var P = X.shape[1];
            var cov = 1.0 / P * np.dot(X, X.T);
            print(cov);

            var (D, V) = np.linalg.eigh(cov);


            print($"特征向量：\r\n{V}");
            print($"特征值：\r\n{D}");

            var c = V[$"{np.argmax(D)}"];
            var encoded = np.matmul(V, X);
            print($"编码\r\n:{encoded}");
            var decoded = np.matmul(c.T, encoded);
            print($"解码\r\n:{decoded}");

            var (U, S, v) = np.linalg.svd(cov);
            print(U);
            print(S);
            print(v);
        }

        /// <summary>
        ///     Based on eigenvalue decomposition
        /// </summary>
        [Fact]
        public void PCA_test()
        {
            var X = np.array(new double[,] {{-1, -1, 0, 2, 0}, {-2, 0, 0, 1, 1}});

            var model = new MLNet.Models.UnSuperviserModel.PCA(0.5);
            model.Fit(X);
            print(model.Resolve);
        }
    }
}