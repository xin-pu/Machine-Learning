using MLNet.LearningModel;
using MLNet.Utils;

namespace MLNet.Regression.LinearRegression
{
    public abstract class LinearRegression
        : LinearModel
    {
        protected LinearRegression(
            string name,
            PrimaryType primaryType = PrimaryType.Polynomial,
            int alpha = 16) :
            base(name, primaryType, alpha)
        {
        }
    }
}