using Numpy;

namespace MLNet.Utils
{
    public class Convert
    {
        public static NDarray ConvertX(double[,] x, int alpha)
        {
            var batch = x.GetLength(0);
            var dims = x.GetLength(1);
            var res = np.zeros(batch, alpha);
            Enumerable.Range(0, batch).ToList().ForEach(i =>
            {
                res[i] = np.array(PrimaryFunc.getPolyPrimary(x[i, 0], alpha));
            });
            return res;
        }
    }
}