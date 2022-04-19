using Numpy;

namespace MLNet.Utils
{
    public static class NumpyExp
    {
        public static NDarray CvtToLinearX(NDarray a)
        {
            var b = np.ones(a.shape[0]);
            var res = np.insert(a, 0, b, 1);
            return res;
        }

        public static NDarray Power(NDarray a, double power)
        {
            var p = np.ones_like(a) * power;
            return np.power(a, p);
        }
    }
}