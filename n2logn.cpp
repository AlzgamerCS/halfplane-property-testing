#include <bits/stdc++.h>
using namespace std;

struct point
{
    double x, y;
    int c;
    point(double x = 0, double y = 0, int c = 0) : x(x), y(y), c(c) {}
    double l() const { return x * x + y * y; }
    point operator+(const point a) { return point(x + a.x, y + a.y, c); }
    point operator-(const point a) { return point(x - a.x, y - a.y, c); }
    double operator*(const point a) const { return x * a.y - y * a.x; }
    double operator%(const point a) const { return x * a.x + y * a.y; }
    bool operator<(const point a)
    {
        if (x == a.x)
            return y < a.y;
        return x < a.x;
    }
    bool operator==(const point a) { return x == a.x && y == a.y; }
    int pos() const
    {
        if (y > 0 || (y == 0 && x > 0))
            return 0;
        return 1;
    }
};

bool cmp(const point &a, const point &b)
{
    if (a.pos() == b.pos())
        return a * b > 0 || (a * b == 0 && a.l() < b.l());
    return a.pos() < b.pos();
}

bool check(const point &a, const point &b)
{
    double v = a * b;
    return v > 0 || (v == 0 && (a % b > 0));
}

int main()
{
    ios_base::sync_with_stdio(0), cin.tie(0);
    int n;
    cin >> n;
    vector<point> p(n);
    int all = 0;
    for (int i = 0; i < n; ++i)
    {
        char c;
        cin >> p[i].x >> p[i].y >> c;
        p[i].c = (c == '#');
        all += p[i].c == 1;
    }
    int ans = n;
    for (int l = 0; l < n; ++l)
    {
        vector<point> v = p;
        int m = n - 1;
        swap(v[m], v[l]);
        for (int j = 0; j < m; ++j)
            v[j] = v[j] - v[m];
        sort(v.begin(), v.end() - 1, cmp);
        /* for (int i = 0; i < m; ++i) { */
        /*     cout << v[i].x << " " << v[i].y << '\n'; */
        /* } */
        /* cout << "___________\n"; */
        int now = all;
        now += (v[m].c == 0) - (v[m].c == 1);
        for (int i = 0, j = -1; i < m; ++i)
        {
            j = max(j, i - 1);
            if (i > 0 && v[i] * v[i - 1] == 0 && v[i] % v[i - 1] > 0)
            {
                now -= (v[i].c == 0) - (v[i].c == 1);
                continue;
            }
            while (j + 2 - i <= m && check(v[i], v[(j + 1) % m]))
            {
                j++;
                now += (v[j % m].c == 0) - (v[j % m].c == 1);
            }
            ans = min(ans, now);
            ans = min(ans, n - now);
            now -= (v[i].c == 0) - (v[i].c == 1);
        }
        now = all;
        now += (v[m].c == 0) - (v[m].c == 1);
        for (int i = 0, j = -1; i < m; ++i)
        {
            j = max(j, i - 1);
            if (i > 0 && v[i] * v[i - 1] == 0 && v[i] % v[i - 1] > 0)
            {
                now -= (v[i].c == 0) - (v[i].c == 1);
                continue;
            }
            while (j + 2 - i <= m && v[i] * v[(j + 1) % m] >= 0)
            {
                j++;
                now += (v[j % m].c == 0) - (v[j % m].c == 1);
            }
            ans = min(ans, now);
            ans = min(ans, n - now);
            now -= (v[i].c == 0) - (v[i].c == 1);
        }
    }
    cout << ans << endl;
}
