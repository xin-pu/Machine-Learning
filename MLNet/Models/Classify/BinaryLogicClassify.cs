using AutoDiff;
using MLNet.Kernels;
using MLNet.Losses;
using MLNet.Utils;
using Numpy;

namespace MLNet.Models.Classify
{
    public class BinaryLogicClassify : Model
    {
        public BinaryLogicClassify()
        {
            Name = "BinaryLogicClassify";
        }

        internal override NDarray transform(NDarray x)
        {
            return new Gaussian(2).Transform(x);
        }


        internal override NDarray call(NDarray x)
        {
            if (Resolve == null) throw new Exception("Resolve is Empty");
            return np2.sigmoid(np.matmul(x, Resolve));
        }

        internal override Loss initialLoss(Variable[] variables, NDarray x, NDarray y)
        {
            return new CrossEntropy(variables, x, y);
        }
    }
}