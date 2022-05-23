using Numpy;
using Numpy.Models;

namespace MLNet.Models.UnSuperviserModel
{
    public class PCA : Model
    {
        public PCA(double componentRatio = 0.99)
        {
            ComponentRatio = componentRatio;
        }


        public double ComponentRatio { protected set; get; }

        public void Fit(NDarray traindatas_x)
        {
            var mean = np.mean(traindatas_x, -1, keepdims: true);
            var X_center = traindatas_x - mean;

            var P = X_center.shape[1];
            var cov = 1.0 / P * np.dot(X_center, X_center.T);


            var (D, V) = np.linalg.eigh(cov);


            var feature = D.GetData<double>();

            var sum = feature.Sum() * ComponentRatio;

            var dict = feature
                .Select((v, i) => new KeyValuePair<int, double>(i, v))
                .OrderBy(p => p.Value)
                .ToDictionary(p => p.Key, p => p.Value);

            var s = 0.0;
            var resolve = new List<NDarray>();
            foreach (var keyValuePair in dict)
            {
                s += keyValuePair.Value;
                if (s > sum)
                    break;
                resolve.Add(V[$"{keyValuePair.Key}"]);
            }

            Resolve = np.stack(resolve.ToArray());
        }


        public override void Fit(NDarray traindatas_x, NDarray traindatas_y, TrainPlan trainPlan)
        {
        }

        public override void InitialWeights(NDarray traindatas_x, NDarray trandatas_y)
        {
        }

        internal override NDarray call(NDarray x, Shape shape)
        {
            return np.matmul(Resolve, x);
        }
    }
}