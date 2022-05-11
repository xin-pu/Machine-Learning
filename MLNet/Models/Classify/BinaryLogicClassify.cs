using AutoDiff;
using MLNet.Utils;
using Numpy;
using Numpy.Models;

namespace MLNet.Models.Classify
{
    public class BinaryLogicClassify : SupervisedModel
    {
        public BinaryLogicClassify()
        {
            Name = "BinaryLogicClassify";
        }


        internal override Variable[] initialVariables(NDarray x, NDarray y)
        {
            var featureCount = x.shape[1];
            var variables = Enumerable.Range(0, featureCount).Select(_ => new Variable()).ToArray();
            return variables;
        }

        internal override NDarray call(NDarray x, Shape shape)
        {
            if (Resolve == null) throw new Exception("Resolve is Empty");
            var y_pred = np2.sigmoid(np.matmul(x, Resolve));
            return np.reshape(y_pred, shape);
        }
    }
}