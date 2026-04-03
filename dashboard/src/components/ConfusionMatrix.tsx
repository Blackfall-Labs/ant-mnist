interface Props {
  confusion: number[][];
}

/** 10x10 confusion matrix with green diagonal, red off-diagonal. */
export function ConfusionMatrix({ confusion }: Props) {
  if (!confusion || confusion.length < 10) return null;

  return (
    <table className="border-collapse text-center" style={{ fontSize: 8 }}>
      <thead>
        <tr>
          <th className="text-gray-600 px-0.5" />
          {Array.from({ length: 10 }, (_, c) => (
            <th key={c} className="text-gray-600 px-0.5 w-5">
              {c}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {confusion.map((row, r) => {
          const rowMax = Math.max(1, ...row);
          return (
            <tr key={r}>
              <th className="text-gray-600 px-0.5">{r}</th>
              {row.map((v, c) => {
                const intensity = v / rowMax;
                const bg =
                  r === c
                    ? `rgba(76,175,80,${intensity})`
                    : `rgba(244,67,54,${intensity * 0.7})`;
                return (
                  <td
                    key={c}
                    className="w-5 h-4 border border-gray-900"
                    style={{ background: bg }}
                    title={`${r}→${c}: ${v}`}
                  >
                    {v || ""}
                  </td>
                );
              })}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
