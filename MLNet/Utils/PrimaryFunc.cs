using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace MLNet.Utils
{
    public class PrimaryFunc
    {
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

        /// <summary>
        ///     多项式奇函数
        ///     return column [1,x,x^2,....x^order]
        /// </summary>
        public static Func<double, int, double[]> getPolyPrimary = (x, order) =>
        {
            return Enumerable.Range(0, order).Select(o => Math.Pow(x, o)).ToArray();
        };


        public static Func<double[], int, Matrix<double>> getTrigPrimaryS = (x, order) =>
        {
            var rows = x.Select(i => getTrigPrimary(i, order));
            return Matrix.Build.DenseOfRowArrays(rows);
        };

        public static Func<double[], int, Matrix<double>> getPolyPrimaryS = (x, order) =>
        {
            var rows = x.Select(i => getPolyPrimary(i, order));
            return Matrix.Build.DenseOfRowArrays(rows);
        };
    }

    public enum PrimaryType
    {
        Polynomial,
        Triangle
    }
}