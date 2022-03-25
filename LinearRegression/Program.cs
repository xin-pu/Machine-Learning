// See https://aka.ms/new-console-template for more information

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Random;

TestLinearModel();
TestKernelModel();
Console.ReadKey();

void TestLinearModel()
{
    var datas = generateXY(-3, 3, 50);
    var x = datas.Item1;
    var y = datas.Item2;
    var rows = x.ToList().Select(getTrianglePhi).ToList();
    var X = Matrix.Build.DenseOfRowArrays(rows);
    var Y = Matrix.Build.DenseOfColumnArrays(y);
    var res = X.Svd().Solve(Y);
    var predictY = (X * res).ToColumnMajorArray();

    var loss = y.Zip(predictY, (a, b) => a - b).Sum();
    Console.WriteLine($"LinearModel loss:\t{loss}");
}

void TestKernelModel()
{
    var datas = generateXY(-3, 3, 50);
    var x = datas.Item1;
    var y = datas.Item2;
    var K = raisingDimsByKernel(x);
    var Y = Matrix.Build.DenseOfColumnArrays(y);
    var res = K.LU().Solve(Y);

    var predictY = (K * res).ToColumnMajorArray();
    var loss = y.Zip(predictY, (a, b) => a - b).Sum();
    Console.WriteLine($"KernelModel loss:\t{loss}");
}

/// <summary>
/// y=sin(x)/x+0.1*x
/// </summary>
Tuple<double[], double[]> generateXY(double xmin, double xmax, int length)
{
    var step = (xmax - xmin) / length;
    var x = Enumerable.Range(0, length).Select(i => xmin + i * step).ToArray();
    var y = x.Select(getY).ToArray();
    return new Tuple<double[], double[]>(x, y);
}

double getY(double x)
{
    return 2 * Math.Sin(2 * x) + Math.Pow(x, 3) + SystemRandomSource.Default.NextDouble() * 0.02;
}

/// <summary>
/// 三角基函数
/// return column [1,sin(x/2),cos(x/2),....sin(15x/2),cos(15x/2)
/// </summary>
double[] getTrianglePhi(double x)
{
    var res = new double[31];
    res[0] = 1;
    Enumerable.Range(1, 15).ToList().ForEach(i =>
    {
        res[i * 2 - 1] = Math.Sin(i * x / 3);
        res[i * 2] = Math.Cos(i * x / 2);
    });

    return res;
}

/// <summary>
/// 通过高斯核升维
/// </summary>
Matrix<double> raisingDimsByKernel(double[] x)
{
    var m = x.Select(x1 => x.Select(x2 => get_kernel(x1, x2, 0.3)).ToArray());
    return Matrix.Build.DenseOfRowArrays(m);
}


double get_kernel(double x, double l, double sigma)
{
    return Math.Exp(-Math.Pow(x - l, 2) / (2 * Math.Pow(sigma, 2)));
}