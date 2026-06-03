"use client"

import { useRef, useTransition } from "react"
import type { RowScanColumn } from "@/lib/queries/apply-row-scan"
import { addRowScanColumnAction, deleteRowScanColumnAction } from "./actions"

interface Props {
  columns: RowScanColumn[]
}

export default function RowScanTable({ columns }: Props) {
  const [isPending, startTransition] = useTransition()
  const addDialogRef = useRef<HTMLDialogElement>(null)

  function handleAdd(formData: FormData) {
    startTransition(async () => {
      await addRowScanColumnAction(formData)
      addDialogRef.current?.close()
    })
  }

  function handleDelete(tableName: string, columnName: string) {
    if (!confirm(`Remove row-scan column "${tableName}.${columnName}"?`)) return
    startTransition(async () => {
      await deleteRowScanColumnAction(tableName, columnName)
    })
  }

  // Group by table name for readability.
  const grouped = columns.reduce<Record<string, RowScanColumn[]>>((acc, c) => {
    ;(acc[c.table_name] ??= []).push(c)
    return acc
  }, {})

  return (
    <>
      {/* Toolbar */}
      <div className="flex items-center justify-between mb-4">
        <p className="text-sm text-base-content/60">
          {columns.length} column{columns.length !== 1 ? "s" : ""} across{" "}
          {Object.keys(grouped).length} table{Object.keys(grouped).length !== 1 ? "s" : ""}
        </p>
        <button
          className="btn btn-primary btn-sm"
          onClick={() => addDialogRef.current?.showModal()}
        >
          + Add column
        </button>
      </div>

      {/* Table */}
      {columns.length === 0 ? (
        <div className="py-12 text-center text-base-content/40 rounded-box border border-base-300">
          No row-scan columns configured.
        </div>
      ) : (
        <div className="space-y-4">
          {Object.entries(grouped).map(([tableName, cols]) => (
            <div key={tableName} className="rounded-box border border-base-300 overflow-hidden">
              <div className="bg-base-200 px-4 py-2 font-mono text-sm font-medium">
                {tableName}
                <span className="badge badge-ghost badge-sm ml-2">{cols.length}</span>
              </div>
              <table className="table table-sm">
                <thead>
                  <tr>
                    <th>Column</th>
                    <th>Added</th>
                    <th className="w-20" />
                  </tr>
                </thead>
                <tbody>
                  {cols.map((c) => (
                    <tr key={c.column_name}>
                      <td className="font-mono text-xs">{c.column_name}</td>
                      <td className="text-xs text-base-content/50 whitespace-nowrap">
                        {new Date(c.created_at).toLocaleDateString("en-GB", {
                          day: "2-digit", month: "short", year: "numeric",
                        })}
                      </td>
                      <td>
                        <button
                          className="btn btn-ghost btn-xs text-error"
                          onClick={() => handleDelete(c.table_name, c.column_name)}
                          disabled={isPending}
                        >
                          Remove
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </div>
      )}

      {/* ── Add dialog ── */}
      <dialog ref={addDialogRef} className="modal">
        <div className="modal-box">
          <h3 className="font-bold text-lg mb-4">Add Row-Scan Column</h3>
          <p className="text-sm text-base-content/60 mb-4">
            The pipeline runs a cell-by-cell Presidio scan on this column, masking
            any PII it detects. Use for free-text columns (notes, descriptions,
            comments) the column-level classifier can&apos;t fully cover.
          </p>
          <form action={handleAdd} className="space-y-3">
            <label className="form-control">
              <div className="label"><span className="label-text">Table name *</span></div>
              <input
                name="table_name"
                required
                placeholder="e.g. trips"
                className="input input-bordered input-sm font-mono"
              />
            </label>
            <label className="form-control">
              <div className="label"><span className="label-text">Column name *</span></div>
              <input
                name="column_name"
                required
                placeholder="e.g. planning_condition"
                className="input input-bordered input-sm font-mono"
              />
            </label>
            <div className="modal-action">
              <button
                type="submit"
                className="btn btn-primary btn-sm"
                disabled={isPending}
              >
                {isPending ? <span className="loading loading-spinner loading-xs" /> : "Add"}
              </button>
              <button
                type="button"
                className="btn btn-ghost btn-sm"
                onClick={() => addDialogRef.current?.close()}
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
        <form method="dialog" className="modal-backdrop"><button>close</button></form>
      </dialog>
    </>
  )
}
