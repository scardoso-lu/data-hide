"use client"

import { useRef, useTransition } from "react"
import type { ColumnExclusion } from "@/lib/queries/exclusions"
import { addExclusionAction, deleteExclusionAction } from "./actions"

interface Props {
  exclusions: ColumnExclusion[]
}

export default function ExclusionTable({ exclusions }: Props) {
  const [isPending, startTransition] = useTransition()
  const addDialogRef = useRef<HTMLDialogElement>(null)

  function handleAdd(formData: FormData) {
    startTransition(async () => {
      await addExclusionAction(formData)
      addDialogRef.current?.close()
    })
  }

  function handleDelete(tableName: string, columnName: string) {
    if (!confirm(`Remove exclusion for "${tableName}.${columnName}"?`)) return
    startTransition(async () => {
      await deleteExclusionAction(tableName, columnName)
    })
  }

  // Group by table name for readability
  const grouped = exclusions.reduce<Record<string, ColumnExclusion[]>>(
    (acc, ex) => {
      ;(acc[ex.table_name] ??= []).push(ex)
      return acc
    },
    {},
  )

  return (
    <>
      {/* Toolbar */}
      <div className="flex items-center justify-between mb-4">
        <p className="text-sm text-base-content/60">
          {exclusions.length} exclusion{exclusions.length !== 1 ? "s" : ""} across{" "}
          {Object.keys(grouped).length} table{Object.keys(grouped).length !== 1 ? "s" : ""}
        </p>
        <button
          className="btn btn-primary btn-sm"
          onClick={() => addDialogRef.current?.showModal()}
        >
          + Add exclusion
        </button>
      </div>

      {/* Table */}
      {exclusions.length === 0 ? (
        <div className="py-12 text-center text-base-content/40 rounded-box border border-base-300">
          No column exclusions configured.
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
                    <th>Reason</th>
                    <th>Added</th>
                    <th className="w-20" />
                  </tr>
                </thead>
                <tbody>
                  {cols.map((ex) => (
                    <tr key={ex.column_name}>
                      <td className="font-mono text-xs">{ex.column_name}</td>
                      <td className="text-sm text-base-content/60">
                        {ex.reason ?? <span className="opacity-30">—</span>}
                      </td>
                      <td className="text-xs text-base-content/50 whitespace-nowrap">
                        {new Date(ex.created_at).toLocaleDateString("en-GB", {
                          day: "2-digit", month: "short", year: "numeric",
                        })}
                      </td>
                      <td>
                        <button
                          className="btn btn-ghost btn-xs text-error"
                          onClick={() => handleDelete(ex.table_name, ex.column_name)}
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
          <h3 className="font-bold text-lg mb-4">Add Column Exclusion</h3>
          <p className="text-sm text-base-content/60 mb-4">
            Excluded columns pass through the pipeline untouched, regardless
            of what the classification tiers detect.
          </p>
          <form action={handleAdd} className="space-y-3">
            <label className="form-control">
              <div className="label"><span className="label-text">Table name *</span></div>
              <input
                name="table_name"
                required
                placeholder="e.g. ivu_db_ops_fdi_raw"
                className="input input-bordered input-sm font-mono"
              />
            </label>
            <label className="form-control">
              <div className="label"><span className="label-text">Column name *</span></div>
              <input
                name="column_name"
                required
                placeholder="e.g. internal_ref"
                className="input input-bordered input-sm font-mono"
              />
            </label>
            <label className="form-control">
              <div className="label"><span className="label-text">Reason</span></div>
              <input
                name="reason"
                placeholder="Why this column should not be anonymised"
                className="input input-bordered input-sm"
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
