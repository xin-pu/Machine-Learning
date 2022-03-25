// See https://aka.ms/new-console-template for more information

using MathNet.Numerics.LinearAlgebra.Double;

Test();

void Test()
{
    var datas = generateXY(-3, 3, 50);
    var x = datas.Item1;
    var y = datas.Item2;
    var rows = x.ToList().Select(get_φtriangle).ToList();
    var X = Matrix.Build.DenseOfRowArrays(rows);
    var Y = Matrix.Build.DenseOfColumnArrays(y);
    var res = X.Svd().Solve(Y);

    var predictY = (X * res).ToColumnMajorArray();
    Console.ReadKey();
}

/// <summary>
/// y=sin(x)/x+0.1*x
/// </summary>
Tuple<double[], double[]> generateXY(double xmin, double xmax, int length)
{
    var step = (xmax - xmin) / length;
    var x = Enumerable.Range(0, length).Select(i => xmin + i * step).ToArray();
    var y = x.Select(i => 2 * Math.Sin(2 * i) + Math.Pow(i, 3))
        .ToArray();
    return new Tuple<double[], double[]>(x, y);
}

/// <summary>
/// return column [1,sin(x/2),cos(x/2),....sin(15x/2),cos(15x/2)
/// </summary>
double[] get_φtriangle(double x)
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