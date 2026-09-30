-- SignSage: per-user translation history.
-- Run once in the Supabase SQL editor (or via the Supabase CLI).
-- Stores text only -- never camera frames or landmark streams (see docs/security).

create table if not exists public.translation_sessions (
    id             uuid primary key default gen_random_uuid(),
    user_id        uuid not null default auth.uid() references auth.users (id) on delete cascade,
    direction      text not null check (direction in ('sign_to_text', 'text_to_sign')),
    input_text     text,
    output_text    text not null default '',
    confidence     real,
    model_version  text not null default '',
    result_type    text not null check (result_type in ('model', 'avatar_gloss', 'failed')),
    created_at     timestamptz not null default now()
);

create index if not exists translation_sessions_user_created_idx
    on public.translation_sessions (user_id, created_at desc);

alter table public.translation_sessions enable row level security;

-- A user can read, insert and delete only their own rows. No UPDATE policy:
-- history entries are immutable.
create policy "own rows: select" on public.translation_sessions
    for select to authenticated using (auth.uid() = user_id);
create policy "own rows: insert" on public.translation_sessions
    for insert to authenticated with check (auth.uid() = user_id);
create policy "own rows: delete" on public.translation_sessions
    for delete to authenticated using (auth.uid() = user_id);
