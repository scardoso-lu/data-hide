"use client"

import { useRef, useTransition, useState } from "react"
import type { ConfigEntry } from "@/lib/queries/config"
import { upsertConfigAction, deleteConfigAction } from "./actions"

interface Props {
  entries: ConfigEntry[]
}

export default function ConfigTable({ entries }: Props) {
  const [isPending, startTransition] = useTransition()
  const [editEntry, setEditEntry] = useState<ConfigEntry | null>(null)

  const addDialogRef = useRef<HTMLDialogElement>(null)
  const editDialogRef = useRef<HTMLDialogElement>(null)

  function openAdd() {
    addDialogRef.current?.showModal()
  }

  function openEdit(entry: ConfigEntry) {
    setEditEntry(entry)
    editDialogRef.current?.showModal()
  }

  function handleAdd(formData: FormData) {
    startTransition(async () => {
      await upsertConfigAction(formData)
      addDialogRef.current?.close()
    })
  }

  function handleEdit(formData: FormData) {
    startTransition(async () => {
      await upsertConfigAction(formData)
      editDialogRef.current?.close()
    })
  }

  function handleDelete(key: string) {
    if (!confirm(`Delete parameter "${key}"? This cannot be undone.`)) return
    startTransition(async () => {
      await deleteConfigAction(key)
    })
  }

  return (
    <>
      {/* Toolbar */}
      <div className="flex items-center justify-between mb-4">
        <p className="text-sm text-base-content/60">
          {entries.length} parameter{entries.length !== 1 ? "s" : ""}
        </p>
        <button className="btn btn-primary btn-sm" onClick={openAdd}>
          + Add parameter
        </button>
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-box border border-base-300">
        <table className="table table-sm table-zebra">
          <thead>
            <tr>
              <th>Key</th>
              <th>Value</th>
              <th>Description</th>
              <th>Updated</th>
              <th className="w-24" />
            </tr>
          </thead>
          <tbody>
            {entries.length === 0 && (
              <tr>
                <td colSpan={5} className="text-center py-10 text-base-content/50">
                  No parameters configured. Add one above.
                </td>
              </tr>
            )}
            {entries.map((entry) => (
              <tr key={entry.key}>
                <td className="font-mono text-xs font-medium">{entry.key}</td>
                <td className="font-mono text-xs">{entry.value}</td>
                <td className="text-sm text-base-content/60 max-w-xs truncate">
                  {entry.description ?? <span className="opacity-30">—</span>}
                </td>
                <td className="text-xs text-base-content/50 whitespace-nowrap">
                  {new Date(entry.updated_at).toLocaleDateString("en-GB", {
                    day: "2-digit", month: "short", year: "numeric",
                  })}
                </td>
                <td>
                  <div className="flex gap-1 justify-end">
                    <button
                      className="btn btn-ghost btn-xs"
                      onClick={() => openEdit(entry)}
                      disabled={isPending}
                    >
                      Edit
                    </button>
                    <button
                      className="btn btn-ghost btn-xs text-error"
                      onClick={() => handleDelete(entry.key)}
                      disabled={isPending}
                    >
                      Delete
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* ── Add dialog ── */}
      <dialog ref={addDialogRef} className="modal">
        <div className="modal-box">
          <h3 className="font-bold text-lg mb-4">Add Parameter</h3>
          <form action={handleAdd} className="space-y-3">
            <label className="form-control">
              <div className="label"><span className="label-text">Key *</span></div>
              <input
                name="key"
                required
                placeholder="e.g. K_ANONYMITY_MIN"
                className="input input-bordered input-sm font-mono"
              />
            </label>
            <label className="form-control">
              <div className="label"><span className="label-text">Value *</span></div>
              <input
                name="value"
                required
                placeholder="e.g. 5"
                className="input input-bordered input-sm"
              />
            </label>
            <label className="form-control">
              <div className="label"><span className="label-text">Description</span></div>
              <input
                name="description"
                placeholder="Optional human-readable description"
                className="input input-bordered input-sm"
              />
            </label>
            <div className="modal-action">
              <button
                type="submit"
                className="btn btn-primary btn-sm"
                disabled={isPending}
              >
                {isPending ? <span className="loading loading-spinner loading-xs" /> : "Save"}
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

      {/* ── Edit dialog ── */}
      <dialog ref={editDialogRef} className="modal">
        <div className="modal-box">
          <h3 className="font-bold text-lg mb-4">Edit Parameter</h3>
          {editEntry && (
            <form action={handleEdit} className="space-y-3">
              <input type="hidden" name="key" value={editEntry.key} />
              <label className="form-control">
                <div className="label"><span className="label-text">Key</span></div>
                <input
                  value={editEntry.key}
                  readOnly
                  className="input input-bordered input-sm font-mono opacity-60"
                />
              </label>
              <label className="form-control">
                <div className="label"><span className="label-text">Value *</span></div>
                <input
                  name="value"
                  required
                  defaultValue={editEntry.value}
                  className="input input-bordered input-sm"
                />
              </label>
              <label className="form-control">
                <div className="label"><span className="label-text">Description</span></div>
                <input
                  name="description"
                  defaultValue={editEntry.description ?? ""}
                  className="input input-bordered input-sm"
                />
              </label>
              <div className="modal-action">
                <button
                  type="submit"
                  className="btn btn-primary btn-sm"
                  disabled={isPending}
                >
                  {isPending ? <span className="loading loading-spinner loading-xs" /> : "Save"}
                </button>
                <button
                  type="button"
                  className="btn btn-ghost btn-sm"
                  onClick={() => editDialogRef.current?.close()}
                >
                  Cancel
                </button>
              </div>
            </form>
          )}
        </div>
        <form method="dialog" className="modal-backdrop"><button>close</button></form>
      </dialog>
    </>
  )
}
