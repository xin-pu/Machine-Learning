namespace MLNet.Utils
{
    public class PrimaryFunc
    {
        #region func

        /// <summary>
        ///     多项式奇函数
        ///     return column [1,x,x^2,....x^order]
        /// </summary>
        public static Func<double, int, double[]> getPolyPrimary = (x, order) =>
        {
            return Enumerable.Range(0, order).Select(o => Math.Pow(x, o)).ToArray();
        };

        /// <summary>
        ///     三角基函数
        ///     return column [1,sin(x/2),cos(x/2),....sin(15x/2),cos(15x/2)
        /// </summary>
        public static Func<double, int, double[]> getTrigPrimary = (x, order) =>
        {
            var res = new double[order * 2 + 1];
            res[0] = 1;
            Enumerable.Range(1, order).ToList().ForEach(i =>
            {
                res[i * 2 - 1] = Math.Sin(i * x / 3);
                res[i * 2] = Math.Cos(i * x / 2);
            });

            return res;
        };

        #endregion
    }

    public enum PrimaryType
    {
        Original,
        Polynomial,
        Triangle
    }

    public enum MultiPrimaryType
    {
        ADD,
        MUL
    }
}